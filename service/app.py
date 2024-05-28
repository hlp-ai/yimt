import io
import os
import tempfile
import uuid
from functools import wraps
from html import unescape

from flask import (Flask, abort, jsonify, render_template, request, send_file, url_for, g)
from scipy.io.wavfile import write
from werkzeug.utils import secure_filename

from extension.files.translate_files import translate_doc, support
from extension.files.translate_html import translate_tag_list
from extension.files.translate_tag import translate_html
from service import remove_translated_files
from service.api_keys import APIKeyDB
from service.asr import AudioRecognizers, amr2wav
from service.mt import Progress, translator_factory
from service.ocr import TextRecognizers
from service.tts import AudioGenerators
from service.utils import get_logger, path_traversal_check, SuspiciousFileOperation, detect_lang

log_service = get_logger(log_filename="service.log", name="service")


class TranslationProgress(Progress):
    def __init__(self):
        self._progress_info = ""

    def report(self, total, done):
        self._progress_info = "{}/{}".format(done, total)
        print(self._progress_info)

    def get_info(self):
        return self._progress_info



def get_upload_dir():
    upload_dir = os.path.join(tempfile.gettempdir(), "yimt-files-translate")

    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)

    return upload_dir


def get_req_api_key():
    if request.is_json:
        json = get_json_dict(request)
        ak = json.get("api_key")
    else:
        ak = request.values.get("api_key")

    return ak


def get_json_dict(request):
    d = request.get_json()
    if not isinstance(d, dict):
        abort(400, description="Invalid JSON format")
    return d


def get_remote_address():
    if request.headers.getlist("X-Forwarded-For"):
        ip = request.headers.getlist("X-Forwarded-For")[0].split(",")[0]
    else:
        ip = request.remote_addr or "127.0.0.1"

    log_service.info("Request from: " + ip)

    return ip


def get_req_limits(default_limit, api_keys_db, multiplier=1):
    req_limit = default_limit

    if api_keys_db:
        api_key = get_req_api_key()

        if api_key:
            db_req_limit = api_keys_db.lookup(api_key)  # get req limit for api key
            if db_req_limit is not None:
                req_limit = db_req_limit * multiplier

    return req_limit


def get_routes_limits(default_req_limit, daily_req_limit, api_keys_db):
    if default_req_limit == -1:
        # TODO: better way?
        default_req_limit = 9999999999999

    def minute_limits():
        return "%s per minute" % get_req_limits(default_req_limit, api_keys_db)

    def daily_limits():
        return "%s per day" % get_req_limits(daily_req_limit, api_keys_db, 1440)

    res = [minute_limits]

    if daily_req_limit > 0:
        res.append(daily_limits)

    return res


def create_app(args):
    app = Flask(__name__)

    if not args.disable_files_translation:  # clean uploaded files periodically
        remove_translated_files.setup(get_upload_dir())

    translators = translator_factory
    text_recognizers = TextRecognizers()
    audio_recognizers = AudioRecognizers()
    audio_generators = AudioGenerators()

    translate_progress = TranslationProgress()

    lang_pairs, from_langs, to_langs, langs_api = translators.support_languages()

    api_keys_db = None

    if args.req_limit > 0 or args.api_keys or args.daily_req_limit > 0:
        print("Applying request limit...")
        api_keys_db = APIKeyDB() if args.api_keys else None

        from flask_limiter import Limiter

        limiter = Limiter(
            app,
            key_func=get_remote_address,
            default_limits=get_routes_limits(args.req_limit, args.daily_req_limit, api_keys_db),
        )
    else:
        from service.utils import NoLimiter

        limiter = NoLimiter()

    def access_check(f):
        """Check API key"""
        @wraps(f)
        def func(*a, **kw):
            if args.api_keys:  # need API key
                ak = get_req_api_key()
                if not ak:
                    abort(403, description="NO API key")
                elif api_keys_db.lookup(ak) is None:
                    abort(403, description="Invalid API key")

            return f(*a, **kw)

        return func

    @app.errorhandler(400)
    def invalid_api(e):
        return jsonify({"error": str(e.description)}), 400

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": str(e.description)}), 500

    @app.errorhandler(429)
    def slow_down_error(e):
        return jsonify({"error": "Slowdown: " + str(e.description)}), 429

    @app.errorhandler(403)
    def denied(e):
        return jsonify({"error": str(e.description)}), 403

    @app.after_request
    def after_request(response):
        response.headers.add("Access-Control-Allow-Origin", "*")  # Allow CORS from anywhere
        response.headers.add(
            "Access-Control-Allow-Headers", "Authorization, Content-Type"
        )
        response.headers.add("Access-Control-Expose-Headers", "Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        response.headers.add("Access-Control-Max-Age", 60 * 60 * 24 * 20)
        return response

    ##############################################################################################
    #
    # Path for Web server
    #
    ##############################################################################################

    @app.route("/")
    @limiter.exempt
    def index():
        if args.disable_web_ui:
            abort(404)

        return render_template('text.html')

    @app.route("/file")
    @limiter.exempt
    def file():
        if args.disable_web_ui:
            abort(404)

        return render_template('file.html')
    
    @app.route('/text')
    @limiter.exempt
    def text():
        if args.disable_web_ui:
            abort(404)

        return render_template('text.html')

    @app.route('/mobile')
    @limiter.exempt
    def mobile():
        if args.disable_web_ui:
            abort(404)

        return render_template('mobile_text.html')

    @app.route('/usage')
    @limiter.exempt
    def usage():
        if args.disable_web_ui:
            abort(404)

        return render_template('usage.html')

    @app.route('/api_usage')
    @limiter.exempt
    def api_usage():
        if args.disable_web_ui:
            abort(404)

        return render_template('api_usage.html')

    ##############################################################################################
    #
    # 对外接口
    #
    ##############################################################################################

    @app.get("/languages")
    @limiter.exempt
    def languages():
        """Retrieve list of supported languages
        No parameter
        :return list of language dictionary
        """
        log_service.info("/languages")
        supported_languages = langs_api
        return jsonify(supported_languages)

    @app.post("/translate")
    @access_check
    def translate():
        """Translate text from a language to another"""
        if request.is_json:  # json data in body of POST method
            json = get_json_dict(request)
            log_service.info("/translate: {}".format(json))
            q = json.get("q")
            source_lang = json.get("source")
            target_lang = json.get("target")
            text_format = json.get("format")
            api_key = json.get("api_key")
        else:  # url data in body of POST method
            q = request.values.get("q")
            source_lang = request.values.get("source")
            target_lang = request.values.get("target")
            text_format = request.values.get("format")
            api_key = request.values.get("api_key")

        if not q:
            abort(400, description="Invalid request: missing q parameter")
        if not source_lang:
            abort(400, description="Invalid request: missing source parameter")
        if not target_lang:
            abort(400, description="Invalid request: missing target parameter")

        if not text_format:
            text_format = "text"

        if text_format not in ["text", "html"]:
            abort(400, description="%s format is not supported" % text_format)

        if not api_key:
            api_key = ""

        if isinstance(q, list):  # 浏览器插件元素列表翻译
            translations = translate_tag_list(q, source_lang, target_lang)
            resp = {
                'translatedText': translations
            }
            return jsonify(resp)

        q = unescape(q)
        q = q.strip()
        if len(q) == 0:
            return jsonify({'translatedText': ""})

        # Check the length of input text
        if args.char_limit != -1:
            chars = len(q)

            if args.char_limit < chars:
                abort(
                    400,
                    description="Invalid request: Request (%d) exceeds character limit (%d)"
                                % (chars, args.char_limit),
                )

        if source_lang == "auto":
            source_lang = detect_lang(q)

        if source_lang not in from_langs:
            abort(400, description="Source language %s is not supported" % source_lang)

        if target_lang not in to_langs:
            abort(400, description="Target language %s is not supported" % target_lang)

        src = q
        lang = source_lang + "-" + target_lang

        translator = translators.get_translator(source_lang, target_lang)
        if translator is None:
            abort(400, description="Language pair %s is not supported" % lang)

        if text_format == "html":
            translation = str(translate_html(translator, src))
        else:
            translation = translator.translate_paragraph(src)

        log_service.info("/translate: " + "&source=" + source_lang + "&target=" + target_lang
                         + "&format=" + text_format + "&api_key=" + api_key)

        resp = {
            'translatedText': translation
        }
        return jsonify(resp)

    @app.post("/ocr")
    # @access_check
    def image2text():
        json_dict = get_json_dict(request)

        image_64_string = json_dict.get("base64")
        format = json_dict.get("format")
        token = json_dict.get("token")
        lang = json_dict.get("lang")

        log_service.info("/ocr: {}".format(lang))

        if not lang:
            abort(400, description="Invalid request: missing lang parameter")
        else:
            if lang == "zh":
                lang = "ch_sim"
        if not image_64_string:
            abort(400, description="Invalid request: missing base64 parameter")

        if not format:
            format = "jpg"

        import base64
        image_data = base64.b64decode(image_64_string)

        # filepath = os.path.join(get_upload_dir(), "decoded_image.{}".format(format))
        filepath = "decoded_image.{}".format(format)

        with open(filepath, "wb") as image_file:
            image_file.write(image_data)

        result = text_recognizers.recognize(filepath, lang)
        print(result)
        if result is None:
            abort(400, description="NO OCR")

        text = ""
        for p in result:
            text += p[-1] + "\n"

        print(text)

        return jsonify({"text": text})

    @app.post("/asr")
    # @access_check
    def audio2text():
        json = get_json_dict(request)

        audio_64_string = json.get("base64")
        format = json.get("format")
        rate = json.get("rate")
        channel = json.get("channel")
        token = json.get("token")
        len = json.get("len")
        lang = json.get("lang")

        log_service.info("/asr: {}".format(lang))

        if not audio_64_string:
            abort(400, description="Invalid request: missing audio base64 parameter")

        import base64
        audio_data = base64.b64decode(audio_64_string)
        temp_audio_file = "temp_audo.{}".format(format)
        with open(temp_audio_file, "wb") as audio_file:
            audio_file.write(audio_data)

        if format == "amr":
            print("转换AMR文件...")
            temp_wav_file = "temp_audo.wav"
            amr2wav(temp_audio_file, temp_wav_file)
            # audio = AudioSegment.from_file(temp_audio_file)
            # # print(len(song)) #时长，单位：毫秒
            # # print(song.frame_rate) #采样频率，单位：赫兹
            # # print(song.sample_width) #量化位数，单位：字节
            # # print(song.channels) #声道数，常见的MP3多是双声道的，声道越多文件也会越大。
            # print(audio.frame_rate, audio.channels)
            # wav = np.array(audio.get_array_of_samples())
            # wav = wav / 32768
            # wav = wav.astype(np.float32)
            result = audio_recognizers.recognize_file(temp_wav_file)
        else:
            result = audio_recognizers.recognize_file(temp_audio_file)

        print(result)
        return jsonify(result[0])

    @app.post("/tts")
    # @access_check
    def text2speech():
        json = get_json_dict(request)
        log_service.info("/tts: {}".format(json))

        token = json.get("token")
        text = json.get("text")
        lang = json.get("lang")

        if not text:
            abort(400, description="Invalid request: missing text parameter")
        if not lang:
            abort(400, description="Invalid request: missing language parameter")

        if lang == "en":
            lang = "eng"
        elif lang == "zh":
            lang = "zho"

        print(lang, text)

        result = audio_generators.generate(text, lang)
        if result is None:
            abort(400, description="NO TTS")

        import base64
        r = result[0]

        print("写临时声音文件...")
        tmp_file = "./temp.wav"
        write(tmp_file, r["sr"], r["audio"])

        audio_64_string = base64.b64encode(open(tmp_file, "rb").read())

        # audio_64_string = base64.b64encode(r["audio"].tobytes())

        return jsonify({
            'base64': audio_64_string.decode('utf-8'),
            'rate': r["sr"],
            "type": "wav"
        })


    @app.post("/translate_file")
    @access_check
    def translate_file():
        """Translate file from a language to another"""
        if args.disable_files_translation:
            abort(403, description="Files translation are disabled on this server.")

        source_lang = request.form.get("source")
        target_lang = request.form.get("target")
        file = request.files['file']

        api_key = request.form.get("api_key")
        if not api_key:
            api_key = ""

        if not file:
            abort(400, description="Invalid request: missing file parameter")
        if not source_lang:
            abort(400, description="Invalid request: missing source parameter")
        if not target_lang:
            abort(400, description="Invalid request: missing target parameter")

        if file.filename == '':
            abort(400, description="Invalid request: empty file")

        log_service.info("/translate_file: " + file.filename + "&source=" + source_lang + "&target=" + target_lang
                         + "&api_key=" + api_key)

        file_type = os.path.splitext(file.filename)[1]

        if not support(file_type):
            abort(400, description="Invalid request: file format not supported")

        try:
            filename = str(uuid.uuid4()) + '.' + secure_filename(file.filename)
            filepath = os.path.join(get_upload_dir(), filename)
            file.save(filepath)

            translated_file_path = translate_doc(filepath, source_lang, target_lang, callbacker=translate_progress)
            translated_filename = os.path.basename(translated_file_path)

            suffix = filepath.split(".")[-1]

            # log_service.info("->Translated: from " + filepath + " to " + translated_filename)

            return jsonify(
                {
                    "translatedFileUrl": url_for('download_file', filename=translated_filename, _external=True),
                    "filepath": filepath,
                    "translated_file_path": translated_file_path,
                    "file_type": suffix
                }
            )
        except Exception as e:
            abort(500, description=e)


    @app.get("/download_file/<string:filename>")
    def download_file(filename: str):
        """Download a translated file"""
        if args.disable_files_translation:
            abort(400, description="Files translation are disabled on this server.")

        filepath = os.path.join(get_upload_dir(), filename)
        try:
            checked_filepath = path_traversal_check(filepath, get_upload_dir())
            if os.path.isfile(checked_filepath):
                filepath = checked_filepath
        except SuspiciousFileOperation:
            abort(400, description="Invalid filename")

        log_service.info("/download_file: " + filepath)

        return_data = io.BytesIO()
        with open(filepath, 'rb') as fo:
            return_data.write(fo.read())
        return_data.seek(0)

        download_filename = filename.split('.')
        download_filename.pop(0)  # remove the prefix generated by system
        download_filename = '.'.join(download_filename)

        return send_file(return_data, as_attachment=True, download_name=download_filename)

    @app.post("/request_ad")
    # @access_check
    def request_ad():
        json = get_json_dict(request)
        log_service.info("/add: {}".format(json))

        platform = json.get("platform")
        support_platforms = ["app", "web", "plugin"]

        if not platform:
            abort(400, description="Invalid request: missing parameter: platform")
        if platform not in support_platforms:
            abort(400, description="platform %s is not supported" % platform)

        ad_id = "AD-20221020"
        if platform == "web" or platform == "app":
            type = "image"
        else:
            type = "text"

        ad_text = "Welcome!\n This is a just test."
        if type == "text":
            content = ad_text
        else:
            import base64
            with open("./static/img/ad11.png", "rb") as image_file:  # 设置本地图片路径
                encoded_image = base64.b64encode(image_file.read())
            image_file.close()
            content = encoded_image.decode('utf-8')

        ad_url = "http://www.hust.edu.cn/"  # for test

        log_service.info("/request_ad: " + "platform=" + platform + "&ad_id=" + ad_id)

        resp = {
            'ad_id': ad_id,
            'type': type,
            'content': content,
            'url': ad_url
        }
        return jsonify(resp)

    #####################################################################
    #
    # 内部路径
    #
    #####################################################################

    @app.route("/reference")
    @limiter.exempt
    def reference():
        if args.disable_web_ui:
            abort(404)
        return render_template('reference.html')


    @app.route("/translate_file_progress", methods=['GET', 'POST'])
    def get_translate_progress():
        file = request.files['file']
        file_type = os.path.splitext(file.filename)[1]
        progress = translate_progress.get_info()

        return progress

    @app.post("/get_blob_file")
    # @access_check
    def get_blob_file():
        json = get_json_dict(request)
        file_path = json.get("file_path")
        # print("get_blob_file()"+file_path)
        # print("get_blob_file_path:" + file_path)  # for test
        import base64
        file_64_string = base64.b64encode(open(file_path, "rb").read())
        # print(file_64_string.decode('utf-8'))  # for test
        resp = {
            'base64': file_64_string.decode('utf-8')
        }
        return jsonify(resp)

    @app.post("/get_download")
    def get_download():
        translate_file_path = request.form.get("translated_file_path")
        # print("download trans_path:" + translate_file_path)  # for test
        return url_for('download_file', filename=os.path.basename(translate_file_path), _external=True)

    @app.get("/pptx_original")
    def pptx_original():
        # print("path_original:")
        file_path = request.args.get('file_path')
        print("pptx_original: " + file_path)
        return send_file(file_path)

    @app.get("/pptx_target")
    def pptx_target():
        # print("path_target:")
        translate_file_path = request.args.get('translated_file_path')
        print("pptx_target: " + translate_file_path)
        return send_file(translate_file_path)

    @app.get("/request_original")
    def request_original():
        # print("tph_original:")
        file_type = request.args.get('file_type')
        file_path = request.args.get('file_path')
        # print("type:"+file_type)
        if file_type == 'docx' or file_type == 'pptx' or file_type == 'xlsx':
            # return send_file("templates/media_original.html")
            file_path_str = url_for('static', filename=file_path)
            file_path_str = file_path_str.replace('/static/', '/')
            file_path_str = file_path_str.lstrip('/')
            # print("file_path_str:" + file_path_str)
            return render_template("media_original.html", file_type=file_type, file_path=file_path_str)
        # print("tph_original:"+ file_path)
        return send_file(file_path)

    @app.get("/request_target")
    def request_target():
        # print("tph_target:")
        file_type = request.args.get('file_type')
        file_path = request.args.get('translated_file_path')
        # print("type:" + file_type)
        if file_type == 'docx' or file_type == 'pptx' or file_type == 'xlsx':
            # return send_file("templates/media_target.html")
            file_path_str = url_for('static', filename=file_path)
            file_path_str = file_path_str.replace('/static/', '/')
            file_path_str = file_path_str.lstrip('/')
            # print("file_path_str:" + file_path_str)
            return render_template("media_target.html", file_type=file_type, file_path=file_path_str)
        # print("tph_target:" + file_path)
        return send_file(file_path)

    return app