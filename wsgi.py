"""生产入口：waitress 启动 Flask 应用。"""
import os

from waitress import serve

from app.web import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    serve(app, host="0.0.0.0", port=port, threads=8)
