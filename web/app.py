import os

from flask import Flask, render_template, request, jsonify, redirect, url_for

from web.tm_utils import TMList

app = Flask(__name__)

TM_DIR = r"D:\kidden\github\yimt\service\tm"

TM_RECORDS = []
FILE = []


@app.route("/")
def index():
    return redirect("/tm")


@app.route("/tm")
def tms():
    tm_files = os.listdir(TM_DIR)
    return render_template("tms.html", tm_files=tm_files)


@app.route("/tm/<string:file>")
def tm(file):
    if len(FILE) == 0 or FILE[0] != file:
        print("OPEN new file", file)
        tm = TMList(os.path.join(TM_DIR, file))

        TM_RECORDS.clear()
        for r in tm.records:
            TM_RECORDS.append(r)

        FILE.clear()
        FILE.append(file)

    return render_template("tm.html", records=TM_RECORDS, file=file)


@app.route("/tm/delete/<int:index>/<string:file>")
def delete(index, file):
    print("DELETE", file, len(TM_RECORDS), index)
    TM_RECORDS.pop(index)

    return redirect(url_for("tm", file=file))
    # return render_template("tm.html", records=TM_RECORDS, file=file)


@app.route("/tm/save/<string:file>")
def save(file):
    print("SAVE", file)
    with open(os.path.join(TM_DIR, file), "w", encoding="utf-8") as f:
        for r in TM_RECORDS:
            f.write(f"<lang-pair>{r['direction']}</lang-pair>\n<source>{r['source']}</source>\n<target>{r['target']}</target>\n\n")

    return redirect(url_for("tm", file=file))


@app.post("/tm/update")
def update():
    r = request.get_json()
    source = r.get("source")
    target = r.get("target")
    index = r.get("index")

    print("UPDATE", source, target, index)

    resp = {
        'data': "ok"
    }
    return jsonify(resp)


if __name__ == "__main__":
    app.run("0.0.0.0", port=8000, debug=True)
