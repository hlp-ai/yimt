import os
import tempfile
from html import unescape

from flask import Flask, abort, jsonify, request

from service.asr import AudioRecognizers
from service.mt import ZhEnJaArTranslator
from service.ocr import TextRecognizers
from service.tts import AudioGenerators


def get_upload_dir():
    upload_dir = os.path.join(tempfile.gettempdir(), "yimt-files-temp")

    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)

    return upload_dir


def get_json_dict(request):
    d = request.get_json()
    if not isinstance(d, dict):
        abort(400, description="Invalid JSON format")
    return d


def run_ocr(image_path, source_lang, queue):
    recognizers = TextRecognizers()
    text = recognizers.recognize(image_path, source_lang)

    queue.put(text)


def create_app():
    app = Flask(__name__)

    text_recognizers = TextRecognizers()
    audio_recognizers = AudioRecognizers()
    audio_generators = AudioGenerators()

    translator = ZhEnJaArTranslator("./infer.yaml")

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

    @app.post("/translate")
    @app.get("/translate")
    def translate():
        """Translate text from a language to another"""
        if request.is_json:  # json data in body of POST method
            json = get_json_dict(request)
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

        # if isinstance(q, list):  # 浏览器插件元素列表翻译
        #     translations = translate_tag_list(q, source_lang, target_lang)
        #     resp = {
        #         'translatedText': translations
        #     }
        #     return jsonify(resp)

        q = unescape(q)
        q = q.strip()
        if len(q) == 0:
            return jsonify({'translation': ""})

        # if source_lang == "auto":
        #     source_lang = detect_lang(q)
        #
        # if source_lang not in from_langs:
        #     abort(400, description="Source language %s is not supported" % source_lang)
        #
        # if target_lang not in to_langs:
        #     abort(400, description="Target language %s is not supported" % target_lang)

        translation = translator.translate_paragraph(q, sl=source_lang, tl=target_lang)

        resp = {
            'translation': translation
        }
        return jsonify(resp)


    @app.post("/ocr")
    # @access_check
    def image2text():
        json_dict = get_json_dict(request)
        image_64_string = json_dict.get("base64")
        token = json_dict.get("token")
        lang = json_dict.get("lang")

        if not lang:
            abort(400, description="Invalid request: missing lang parameter")
        if not image_64_string:
            abort(400, description="Invalid request: missing base64 parameter")

        import base64
        image_data = base64.b64decode(image_64_string)

        filepath = os.path.join(get_upload_dir(), "decoded_image.png")

        with open(filepath, "wb") as image_file:
            image_file.write(image_data)

        result = text_recognizers.recognize(filepath, lang)
        if result is None:
            abort(400, description="NO OCR")

        return jsonify(result)

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

        if not audio_64_string:
            abort(400, description="Invalid request: missing base64 parameter")

        import base64
        audio_data = base64.b64decode(audio_64_string)
        temp_audio_file = "temp_audo.wav"
        with open(temp_audio_file, "wb") as audio_file:
            audio_file.write(audio_data)

        result = audio_recognizers.recognize_file(temp_audio_file)
        return jsonify(result)

    @app.post("/tts")
    # @access_check
    def text2speech():
        json = get_json_dict(request)
        token = json.get("token")
        text = json.get("text")
        lang = json.get("lang")

        if not text:
            abort(400, description="Invalid request: missing text parameter")
        if not lang:
            abort(400, description="Invalid request: missing language parameter")

        result = audio_generators.generate(text, lang)
        if result is None:
            abort(400, description="NO TTS")

        import base64
        output = []
        for r in result:
            audio_64_string = base64.b64encode(r["audio"].tobytes())
            output.append({
                'base64': audio_64_string.decode('utf-8'),
                'rate': r["sr"]
            })

        return jsonify(output)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5555)
