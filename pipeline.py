import pdfplumber
import docx
import re
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

def load_document(file_path):
    text = ""
    file_type = "txt"

    if file_path.endswith(".pdf"):
        file_type = "pdf"
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            text = f"Lỗi đọc PDF: {e}"
            
    elif file_path.endswith(".docx"):
        file_type = "docx"
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
        except Exception as e:
            text = f"Lỗi đọc DOCX: {e}"
            
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            text = "Định dạng không được hỗ trợ hoặc không thể đọc."
            
    return text, file_type

def normalize_text(text):
    # Thay thế nhiều khoảng trắng hoặc newline liên tiếp
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# ===============================
# Chunk theo điều khoản (Line by line parsing)
# ===============================
def chunk_document(text):
    
    lines = text.split('\n')
    chunks = []
    
    current_title = "Mở đầu"
    current_content = []
    
    # Regex bắt đầu: Điều 1., 1., 1.1., Chương I, I., a), b., -, +, *, • v.v.
    # Nhận diện cả chữ hoa/thường, số La Mã, và các ký tự đặc biệt đầu dòng
    title_pattern = re.compile(
        r"^\s*(?:Điều\s+\d+|Chương\s+[IVX\d]+|Phần\s+[IVX\d]+|(?:[IVX]+)[.:]|\d{1,2}(?:\.\d{1,2})*[.:]?|[a-zA-ZđĐ][.)]|[-+*•>])(?:\s|$)",
        re.IGNORECASE | re.UNICODE
    )

    for line in lines:
        if not line.strip():
            continue
        
        match = title_pattern.match(line)
        if match:
            # Lưu chunk cũ nếu có
            if current_content:
                chunks.append({
                    "content": current_title + "\n" + "\n".join(current_content),
                    "metadata": {"type": current_title.strip()}
                })
            current_title = line.strip()
            current_content = []
        else:
            current_content.append(line.strip())

    # Thêm chunk cuối
    if current_content or current_title != "Mở đầu":
         chunks.append({
             "content": (current_title + "\n" + "\n".join(current_content) if current_title != "Mở đầu" else "\n".join(current_content)),
             "metadata": {"type": current_title.strip()}
         })

    # fallback
    if len(chunks) == 0:
        print("⚠ Không tìm thấy title header → fallback")

        chunk_size = 500
        for i in range(0, len(text), chunk_size):
            chunks.append({
                "content": text[i:i+chunk_size],
                "metadata": {"type": "length_split"}
            })

    return chunks