import argparse

from service.app import create_app

DEFAULT_ARGS = {
    'HOST': '127.0.0.1',
    'PORT': 5555,
    'CHAR_LIMIT': -1,
    'REQ_LIMIT': -1,
    'DAILY_REQ_LIMIT': -1,
    'BATCH_LIMIT': 64,
    'DEBUG': True,
    'API_KEYS': False,
    'DISABLE_WEB_UI': False,
    'DISABLE_FILES_TRANSLATION': False,
}


def get_args():
    parser = argparse.ArgumentParser(
        description="YiMT翻译服务器"
    )
    parser.add_argument(
        "--host", type=str, help="翻译服务运行主机(%(default)s)", default=DEFAULT_ARGS['HOST']
    )
    parser.add_argument("--port", type=int, help="翻译服务端口(%(default)s)", default=DEFAULT_ARGS['PORT'])
    parser.add_argument(
        "--char-limit",
        default=DEFAULT_ARGS['CHAR_LIMIT'],
        type=int,
        metavar="<number of characters>",
        help="翻译请求最大字符长度(%(default)s)",
    )
    parser.add_argument(
        "--req-limit",
        default=DEFAULT_ARGS['REQ_LIMIT'],
        type=int,
        metavar="<number>",
        help="每个客户每分钟最大翻译请求数(%(default)s)",
    )
    parser.add_argument(
        "--daily-req-limit",
        default=DEFAULT_ARGS['DAILY_REQ_LIMIT'],
        type=int,
        metavar="<number>",
        help="每个客户每天最大翻译请求数(%(default)s)",
    )
    parser.add_argument(
        "--api-keys",
        default=DEFAULT_ARGS['API_KEYS'],
        action="store_true",
        help="激活用户API数据库配置信息",
    )
    parser.add_argument(
        "--disable-web-ui", default=DEFAULT_ARGS['DISABLE_WEB_UI'], action="store_true", help="关闭翻译Web界面"
    )
    parser.add_argument(
        "--disable-files-translation", default=DEFAULT_ARGS['DISABLE_FILES_TRANSLATION'], action="store_true",
        help="关闭文档翻译"
    )

    return parser.parse_args()


def main():
    args = get_args()
    app = create_app(args)
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
