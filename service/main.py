import io
import json
import os
import tempfile
import uuid
from functools import wraps
from html import unescape

from flask import (Flask, abort, jsonify, render_template, request, send_file, url_for, g)
from werkzeug.utils import secure_filename

from service.ocr import TextRecognizers


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

    recognizers = TextRecognizers()

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

        result = recognizers.recognize(filepath, lang)
        if result is None:
            abort(400, description="NO OCR")

        return jsonify(result)

        # str = json.dumps(result)
        #
        # return jsonify({"result":str})

    # @app.post("/asr")
    # # @access_check
    # def translate_audio2text():
    #     json = get_json_dict(request)
    #     audio_64_string = json.get("base64")
    #     format = json.get("format")
    #     rate = json.get("rate")
    #     channel = json.get("channel")
    #     token = json.get("token")
    #     len = json.get("len")
    #     source_lang = json.get("source")
    #     target_lang = json.get("target")
    #
    #     from_audio_formats = ["pcm", "wav", "amr", "m4a"]
    #     q = "audio2text"  # for test
    #
    #     if not format:
    #         abort(400, description="Invalid request: missing format parameter")
    #     if not audio_64_string:
    #         abort(400, description="Invalid request: missing base64 parameter")
    #     if not rate:
    #         abort(400, description="Invalid request: missing rate parameter")
    #     if not channel:
    #         abort(400, description="Invalid request: missing channel parameter")
    #     if not len:
    #         abort(400, description="Invalid request: missing len parameter")
    #     if not source_lang:
    #         abort(400, description="Invalid request: missing source parameter")
    #     if not target_lang:
    #         abort(400, description="Invalid request: missing target parameter")
    #     if source_lang == "auto":
    #         source_lang = detect_lang(q)
    #     if source_lang not in from_langs:
    #         abort(400, description="Source language %s is not supported" % source_lang)
    #     if target_lang not in to_langs:
    #         abort(400, description="Target language %s is not supported" % target_lang)
    #
    #     if format not in from_audio_formats:
    #         abort(400, description="Audio format %s is not supported" % format)
    #
    #     import base64
    #     audio_data = base64.b64decode(audio_64_string)
    #     with open("decoded_audio.wav", "wb") as audio_file:
    #         audio_file.write(audio_data)
    #     resp = {
    #         'translatedText': "test text for 'audio to text' "
    #     }
    #     return jsonify(resp)
    #
    # @app.post("/tts")
    # # @access_check
    # def text2speech():
    #     json = get_json_dict(request)
    #     token = json.get("token")
    #     text = json.get("text")
    #     source_lang = json.get("lang")
    #
    #     if not text:
    #         abort(400, description="Invalid request: missing text parameter")
    #     if not source_lang:
    #         abort(400, description="Invalid request: missing source language parameter")
    #     if source_lang == "auto":
    #         source_lang = detect_lang(text)
    #     # if source_lang not in from_langs:
    #     #     abort(400, description="Source language %s is not supported" % source_lang)
    #
    #     result = tts_fn(text, source_lang)
    #     if result is None:
    #         abort(400, description="NO TTS")
    #
    #     import base64
    #     audio_64_string = base64.b64encode(result[0].numpy().tobytes())
    #     resp = {
    #         'base64': audio_64_string.decode('utf-8'),
    #         'rate': result[1]
    #     }
    #     return jsonify(resp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=6666)
