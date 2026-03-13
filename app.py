from flask import Flask, request, jsonify, render_template
from rag import ask, ask_deep
from config import DOCUMENTS_DIR
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"message": "No files received"}), 400

    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        file.save(filepath)

    from ingest import load_documents, split_documents, create_vector_store
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)

    return jsonify({"message": f"{len(files)} file(s) uploaded and processed successfully."})

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Please provide a question"}), 400

    question = data["question"]
    mode = data.get("mode", "standard")

    if mode not in ["standard", "understand", "exam", "practical"]:
        return jsonify({"error": "Mode must be standard, understand, exam or practical"}), 400

    print(f"Question: {question}")
    print(f"Mode: {mode}")

    result = ask(question, mode)

    return jsonify({
        "question": question,
        "mode": mode,
        "answer": result["answer"],
        "sources": result["sources"]
    })

@app.route("/ask/deep", methods=["POST"])
def ask_deep_question():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Please provide a question"}), 400

    question = data["question"]

    print(f"Deep mode question: {question}")

    result = ask_deep(question)

    return jsonify({
        "question": question,
        "mode": "deep",
        "answer": result["answer"],
        "gaps_identified": result["gaps_identified"],
        "web_supplemented": result["web_supplemented"],
        "web_sources": result["web_sources"],
        "sources": result["sources"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "KU-Assist is running"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)