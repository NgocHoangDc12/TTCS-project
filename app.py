import sys
import os
import subprocess
import difflib

try:
    import streamlit as st
except ModuleNotFoundError:
    correct_python = os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe")
    if os.path.exists(correct_python) and sys.executable != correct_python:
        print("\n[AI] Đang tự động chuyển hướng môi trường Python cho bạn vì VS Code đang chọn sai...")
        subprocess.run([correct_python, "-m", "streamlit", "run", __file__])
        sys.exit(0)
    else:
        raise
import os
import shutil
import time
import sys
import json

from pipeline import load_document, normalize_text, chunk_document
from retrieval import Retriever
from reranker import rerank
from llm import compare_clause

DOCUMENTS_FOLDER = "documents"

def extract_meaningful_diff(text_a, text_b):
    """Dùng difflib để trích xuất CHỈ những dòng thực sự khác biệt giữa 2 đoạn text.
    Trả về (diff_a, diff_b) chỉ chứa các dòng có sự thay đổi.
    Nếu không có khác biệt, trả về (None, None)."""
    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()
    
    diff_lines_a = []
    diff_lines_b = []
    
    matcher = difflib.SequenceMatcher(None, lines_a, lines_b)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            diff_lines_a.extend(lines_a[i1:i2])
            diff_lines_b.extend(lines_b[j1:j2])
        elif tag == 'delete':
            diff_lines_a.extend(lines_a[i1:i2])
        elif tag == 'insert':
            diff_lines_b.extend(lines_b[j1:j2])
        # 'equal' → bỏ qua, không cần gửi cho LLM
    
    if not diff_lines_a and not diff_lines_b:
        return None, None  # Không có khác biệt thực sự
    
    return "\n".join(diff_lines_a), "\n".join(diff_lines_b)

def clear_and_save_files(file_a, file_b):
    """Xóa tài liệu cũ và lưu 2 tài liệu mới tải lên."""
    if os.path.exists(DOCUMENTS_FOLDER):
        shutil.rmtree(DOCUMENTS_FOLDER)
    os.makedirs(DOCUMENTS_FOLDER)

    path_a = os.path.join(DOCUMENTS_FOLDER, file_a.name)
    path_b = os.path.join(DOCUMENTS_FOLDER, file_b.name)

    with open(path_a, "wb") as f:
        f.write(file_a.getbuffer())
    with open(path_b, "wb") as f:
        f.write(file_b.getbuffer())

    return path_a, path_b

# ===============================
# GIAO DIỆN WEB
# ===============================
st.set_page_config(page_title="So Sánh Hợp Đồng", layout="wide")
st.title("Phần mềm So sánh Hợp đồng (LLaMA 3 8B)")

# Khu vực upload 2 file
col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Tải lên Hợp đồng A (Bản gốc)", type=["pdf", "docx"])
with col2:
    file_b = st.file_uploader("Tải lên Hợp đồng B (Bản so sánh)", type=["pdf", "docx"])

if st.button("Bắt đầu so sánh", type="primary"):
    if file_a and file_b:
        with st.spinner("Đang đọc tài liệu và khởi tạo mô hình..."):
            
            # 1. Xóa cũ, lưu mới
            path_a, path_b = clear_and_save_files(file_a, file_b)
            
            # 2. Load và Normalize
            contractA, _ = load_document(path_a)
            contractB, _ = load_document(path_b)
            
            contractA = normalize_text(contractA)
            contractB = normalize_text(contractB)
            
            # 3. Chunking
            chunks_A = chunk_document(contractA)
            chunks_B = chunk_document(contractB)
            
            # 4. Build Retriever cho Hợp đồng B
            retriever_B = Retriever(chunks_B)
            
            st.success(f"Đã xử lý xong! Bản A có {len(chunks_A)} chunk. Bản B có {len(chunks_B)} chunk. Đang đối chiếu...")
            
            st.subheader("Kết quả phân tích chi tiết")
            
            # --- BIPARTITE MATCHING ĐỂ LỌC CÁC ĐOẠN BỊ XÓA & THÊM MỚI ---
            edges = []
            max_A = max(1, len(chunks_A))
            max_B = max(1, len(chunks_B))
            
            for i, clause in enumerate(chunks_A):
                results = retriever_B.retrieve(clause["content"], top_k=10)
                final_results = rerank(clause["content"], results, top_k=10)
                for r in final_results:
                    base_score = r.get("rerank_score", 0)
                    
                    # Tính toán Relative Position Penalty để giữ đúng cấu trúc tuyến tính của hợp đồng
                    pos_A = i / max_A
                    pos_B = r["id"] / max_B
                    penalty = abs(pos_A - pos_B) * 2.0
                    
                    edges.append({
                        "score": base_score - penalty,
                        "raw_score": base_score,
                        "A_idx": i,
                        "B_idx": r["id"],
                    })
                    
            # Sắp xếp để ưu tiên kết nối các cặp giống nhau nhất (đã trừ penalty)
            edges.sort(key=lambda x: x["score"], reverse=True)
            
            matched_A = set()
            matched_B = set()
            match_pairs = []
            
            for edge in edges:
                if edge["A_idx"] not in matched_A and edge["B_idx"] not in matched_B:
                    # Ngưỡng (Threshold) -2.0 cho CrossEncoder kiểm tra trên điểm raw
                    if edge["raw_score"] > -2.0:
                        matched_A.add(edge["A_idx"])
                        matched_B.add(edge["B_idx"])
                        match_pairs.append(edge)
            
            diff_counter = 0
            json_export_data = [] # Lưu toàn bộ kết quả phân tích

            # 5.1. Xử lý các cặp Khớp (Có liên quan - Gọi LLM)
            match_pairs.sort(key=lambda x: x["A_idx"])
            for edge in match_pairs:
                clause_A_content = chunks_A[edge["A_idx"]]["content"]
                clause_B_content = chunks_B[edge["B_idx"]]["content"]
                
                # Bỏ qua hoàn toàn những đoạn giống hệt nhau
                if clause_A_content.replace(" ", "").lower() == clause_B_content.replace(" ", "").lower():
                    continue
                
                # Dùng difflib để trích xuất CHỈ phần thực sự khác biệt
                diff_a, diff_b = extract_meaningful_diff(clause_A_content, clause_B_content)
                
                if diff_a is None and diff_b is None:
                    continue  # difflib xác nhận không có khác biệt thực sự
                
                diff_counter += 1
                
                # Gửi phần khác biệt (đã lọc) cho LLM thay vì toàn bộ chunk
                focused_a = diff_a if diff_a else "(Không có nội dung bị thay đổi)"
                focused_b = diff_b if diff_b else "(Không có nội dung bị thay đổi)"
                answer = compare_clause(focused_a, focused_b)
                
                json_export_data.append({
                    "type": "matched",
                    "clause_A": clause_A_content,
                    "clause_B": clause_B_content,
                    "analysis": answer
                })
                
                clause_title = chunks_A[edge["A_idx"]].get("metadata", {}).get("type", f"Mục {edge['A_idx']+1}")
                with st.expander(f"Phân tích {diff_counter} ({clause_title})", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Nội dung Bản A:**")
                        st.info(clause_A_content)
                    with c2:
                        st.markdown("**Nội dung Bản B:**")
                        st.warning(clause_B_content)
                        
                    st.markdown("**Đánh giá từ LLM:**")
                    st.markdown(answer, unsafe_allow_html=True)

            # 5.2. Các mục bị XÓA hoàn toàn khỏi B (A dư)
            for i, clause in enumerate(chunks_A):
                if i not in matched_A:
                    diff_counter += 1
                    clause_A_content = clause["content"]
                    
                    # Generate Hardcoded Markdown HTML để tiết kiệm API và đảm bảo đúng Format
                    clause_raw = clause_A_content.replace("|", r"\|")
                    answer = f"| Phân loại | Chi tiết thay đổi | Bằng chứng (Trích dẫn gốc) |\n| :--- | :--- | :--- |\n| Lược bỏ nội dung | Nội dung điều luật này trong Bản A đã bị bãi bỏ hoàn toàn trong Bản B | Bản A: {clause_raw} <br> Bản B: Không có nội dung tương ứng. |"
                    
                    json_export_data.append({
                        "type": "deleted",
                        "clause_A": clause_A_content,
                        "clause_B": "Không có nội dung tương ứng.",
                        "analysis": answer
                    })
                    
                    clause_title = clause.get("metadata", {}).get("type", f"Mục {i+1}")
                    with st.expander(f"Phân tích {diff_counter} ({clause_title} - Đã bị bãi bỏ)", expanded=False):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Nội dung Bản A:**")
                            st.info(clause_A_content)
                        with c2:
                            st.markdown("**Nội dung Bản B:**")
                            st.warning("Không có nội dung tương ứng.")
                        st.markdown("**Đánh giá hệ thống:**")
                        st.markdown(answer, unsafe_allow_html=True)
                        
            # 5.3. Các mục được THÊM MỚI vào B (B dư)
            for j, clause in enumerate(chunks_B):
                if j not in matched_B:
                    diff_counter += 1
                    clause_B_content = clause["content"]
                    
                    clause_raw = clause_B_content.replace("|", r"\|")
                    answer = f"| Phân loại | Chi tiết thay đổi | Bằng chứng (Trích dẫn gốc) |\n| :--- | :--- | :--- |\n| Bổ sung nội dung | Nội dung điều luật mới hoàn toàn được bổ sung vào Bản B | Bản A: Không có nội dung tương ứng. <br> Bản B: {clause_raw} |"
                    
                    json_export_data.append({
                        "type": "added",
                        "clause_A": "Không có nội dung tương ứng.",
                        "clause_B": clause_B_content,
                        "analysis": answer
                    })
                    
                    clause_title = clause.get("metadata", {}).get("type", f"Mục mới (B)")
                    with st.expander(f"Phân tích {diff_counter} (Bản B: {clause_title} - Bổ sung mới)", expanded=False):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Nội dung Bản A:**")
                            st.info("Không có nội dung tương ứng.")
                        with c2:
                            st.markdown("**Nội dung Bản B:**")
                            st.warning(clause_B_content)
                        st.markdown("**Đánh giá hệ thống:**")
                        st.markdown(answer, unsafe_allow_html=True)
            
            # --- XUẤT KẾT QUẢ RA FILE JSON ---
            st.divider()
            st.success("✅ Đã hoàn thành quá trình trích xuất và so sánh!")
            
            # LƯU FILE LOCAL
            export_path = os.path.join(DOCUMENTS_FOLDER, "comparison_results.json")
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(json_export_data, f, ensure_ascii=False, indent=4)
                
            # TẠO NÚT DOWNLOAD CHO NGƯỜI DÙNG STREAMLIT
            json_string = json.dumps(json_export_data, ensure_ascii=False, indent=4)
            st.download_button(
                label="📥 Tải file JSON kết quả xuống máy",
                file_name="comparison_results.json",
                mime="application/json",
                data=json_string
            )
                    
    else:
        st.error("⚠️ Vui lòng tải lên đủ cả 2 file hợp đồng!")

if __name__ == '__main__':
    # Kiểm tra xem có đang chạy trong nền Streamlit không (Tránh Infinite Loop)
    if not st.runtime.exists():
        import sys
        from streamlit.web import cli as stcli
        
        # Mồi lệnh tự động gọi Streamlit 
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())