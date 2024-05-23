from onmt.translate import TranslationServer

config_file = "./nllb.conf.json"
translation_server = TranslationServer()
translation_server.start(config_file)

inputs = [{"id": 100,
         "src": "This is a test."}]
trans, scores, n_best, _, aligns = translation_server.run(inputs)