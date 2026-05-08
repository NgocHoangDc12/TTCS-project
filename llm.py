import requests
import re
import difflib

LMSTUDIO_API = "http://localhost:1234/v1/chat/completions"

def compare_clause(clause_A_clean, clause_B_clean):
    system_message = "Bạn là một robot so sánh văn bản. YÊU CẦU TỐI THƯỢNG: Chỉ được phép trả về một BẢNG MARKDOWN duy nhất. KHÔNG giải thích, KHÔNG thêm tiêu đề, KHÔNG thêm ghi chú bên ngoài bảng. BẮT BUỘC XUỐNG DÒNG khi kết thúc một hàng của bảng."
    
    prompt = f"""Nhiệm vụ: Đối chiếu và tìm ra SỰ KHÁC BIỆT THỰC SỰ giữa Bản A và Bản B. BỎ QUA NHỮNG PHẦN GIỐNG NHAU HOÀN TOÀN.
CẢNH BÁO TỔNG HỢP: BẠN PHẢI GỘP TẤT CẢ SỰ THAY ĐỔI VÀO CÙNG 1 Ý (1 DÒNG DUY NHẤT). TUYỆT ĐỐI KHÔNG duyệt qua từng cụm từ hay từng dòng để tạo ra nhiều hàng lắt nhắt. Nếu đoạn cuối giống nhau thì tuyệt đối không được phép bịa ra khác biệt.

QUY TẮC SO SÁNH:
1. PHÂN LOẠI KHÁC BIỆT: Bạn BẮT BUỘC PHẢI COPY CHÍNH XÁC 100% một trong 5 Tên Nhóm dưới đây (Không được sai chính tả, không được tự chế tên lạ) để điền vào cột Phân loại:
   - "Thay đổi Thông số/Số liệu": Sự thay đổi về giá trị định lượng như ngày tháng, thời hạn, số tiền, tỷ lệ %, thông tin định danh cá nhân/số tài khoản (Ví dụ: [Giá trị A] đổi thành [Giá trị B]).
   - "Thay đổi Bản chất pháp lý": Đổi từ ngữ làm biến đổi hoàn toàn ý nghĩa, quyền, nghĩa vụ, đối tượng hoặc trách nhiệm cốt lõi (Ví dụ: [Từ mang tính tùy chọn "có quyền"] đổi thành [Từ mang tính bắt buộc "phải"]).
   - "Chỉnh lý câu chữ": Rút gọn câu, dùng từ đồng nghĩa, hoặc sửa cách diễn đạt cho mượt mà nhưng KHÔNG làm thay đổi bản chất cốt lõi của quyền và nghĩa vụ quy định.
   - "Lược bỏ nội dung": Một phần thông tin, quyền lợi, hay điều kiện có ở Bản A nhưng bị xóa mất ở Bản B.
   - "Bổ sung nội dung": Một phần thông tin, quyền lợi, hoặc điều kiện hoàn toàn mới được chêm vào Bản B.

2. ĐẶC BIỆT XỬ LÝ TRƯỜNG HỢP LỆCH NỘI DUNG VÀ KHÔNG CÙNG QUY ĐỊNH:
   - Hệ thống tự động trích xuất nội dung có thể làm 2 đoạn KHÔNG LIÊN QUAN ĐẾN NHAU bị đặt cạnh nhau (Ví dụ: Bản A là "Thiết bị làm việc", Bản B lại là "Bộ phận công tác").
   - Nếu 2 đoạn hoàn toàn KHÁC CẢ SỐ THỨ TỰ LẪN NGỮ CẢNH (Ví dụ: Bản A điều 2.3 là "Thiết bị", Bản B điều 1.3 là "Bộ phận"): TUYỆT ĐỐI KHÔNG ráng ép so sánh. Đây là trường hợp Bản A bị xóa và Bản B được thêm mới. Bạn BẮT BUỘC tạo 2 hàng độc lập: 
      + Hàng 1: "Lược bỏ nội dung" (Bằng chứng: Bản A copy nguyên văn <br> Bản B ghi "Không có nội dung tương ứng"). 
      + Hàng 2: "Bổ sung nội dung" (Bằng chứng: Bản A ghi "Không có nội dung tương ứng" <br> Bản B copy nguyên văn).
   - NGƯỢC LẠI, NẾU HAI BẢN CÓ CÙNG SỐ THỨ TỰ/ĐẦU MỤC (Ví dụ: Cùng là "Điều 2: ...") HOẶC CÓ CHUNG MỘT CHỦ ĐỀ: Chúng mặc định đang nói về cùng một VẤN ĐỀ. TUYỆT ĐỐI CẤM gán nhãn "Lược bỏ nội dung". Bạn BẮT BUỘC phải dùng nhãn "Thay đổi Thông số/Số liệu", "Thay đổi Bản chất pháp lý" hoặc "Chỉnh lý câu chữ" để thể hiện sự thay đổi cấu trúc/từ ngữ đó.

3. QUY ĐỊNH ĐẦU RA KẾT QUẢ (CHỐNG LỖI BẢNG):
   - CẤM VIẾT BẤT KỲ VĂN BẢN NÀO NGOÀI BẢNG.
   - BẢNG CHỈ ĐƯỢC CÓ ĐÚNG 1 HÀNG DUY NHẤT (Ngoại trừ trường hợp 2 đoạn không liên quan gì nhau theo ngoại lệ ở trên).
   - CỘT "Phân loại" CHỈ ĐƯỢC PHÉP CHỨA ĐÚNG TÊN CỦA 1 TRONG 5 NHÓM ĐÃ NÊU. TUYỆT ĐỐI CẤM TỰ BỊA "dùng từ khác", "xì/viết lại câu", hay bất cứ từ nào sai chính tả.

4. YÊU CẦU CỘT "BẰNG CHỨNG" (TẬP TRUNG CHÍNH XÁC VÀO PHẦN THAY ĐỔI):
   - Cột "Chi tiết thay đổi": Mô tả ngắn gọn sự khác biệt.
   - Cột "Bằng chứng": BẮT BUỘC COPY-PASTE ĐÚNG CHÍNH TẢ phần chữ có sự thay đổi.
   - ĐƯỢC PHÉP VÀ KHUYẾN KHÍCH dùng dấu ba chấm (...) để rút gọn các phần không thay đổi ở đầu/cuối câu. Chỉ tập trung trích dẫn ĐÚNG cụm từ chứa sự khác biệt để tránh lan man, lạc đề (Ví dụ: Bản A: "... bán/mua bất động sản do Bên B là **chủ sở hữu**." <br> Bản B: "... mua/bán bất động sản **thuộc quyền sở hữu** của Bên B.").
   - BẮT BUỘC HIGHLIGHT CHỖ LỖI: Những từ/cụm từ bị đổi PHẢI ĐƯỢC IN ĐẬM bằng cú pháp Markdown (Ví dụ: Bản A: "...thời hạn là **15** ngày..." <br> Bản B: "...thời hạn là **30** ngày...").
   - Bắt buộc dùng thẻ HTML `<br>` để ngắt dòng Bản A và Bản B thành 2 dòng tách biệt trong cùng một ô.

5. YÊU CẦU NGÔN NGỮ VÀ ĐỊNH DẠNG TỐI CAO:
   - CHỈ ĐƯỢC TRẢ LỜI BẰNG TIẾNG VIỆT 100%.
   - KẾT QUẢ CỦA BẠN BẮT BUỘC PHẢI KHỞI ĐẦU VÀ KẾT THÚC BẰNG BẢNG.

--- DỮ LIỆU ĐẦU VÀO ---
[BẢN A]:
{clause_A_clean}

[BẢN B]:
{clause_B_clean}
-----------------------

XUẤT KẾT QUẢ LÀ BẢNG DƯỚI ĐÂY (Vào thẳng vấn đề, cấm nói câu nào trước và sau bảng, Cột Phân loại CẤM sai chính tả):

| Phân loại | Chi tiết thay đổi | Bằng chứng (Trích dẫn gốc) |
| :--- | :--- | :--- |
| [CHÉP ĐÚNG TÊN 1 TRONG 5 NHÓM] | [Mô tả cụ thể] | Bản A: [Trích dẫn] <br> Bản B: [Trích dẫn] |
"""

    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,  
        "max_tokens": 800
    }

    try:
        response = requests.post(LMSTUDIO_API, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "choices" not in result:
            return "⚠️ Lỗi: API không trả về kết quả hợp lệ."

        answer = result["choices"][0]["message"]["content"].strip()

        # Dùng code Python ép cắt bỏ mọi câu chào hỏi/kết luận tiếng Anh của AI
        if "|" in answer:
            start_idx = answer.find("|")
            end_idx = answer.rfind("|") + 1
            answer = answer[start_idx:end_idx].strip()

        return answer
        
    except requests.exceptions.ConnectionError:
        return "⚠️ Lỗi kết nối: Vui lòng kiểm tra xem LM Studio đã được bật 'Start Server' ở port 1234 chưa."
    except Exception as e:
        return f"⚠️ Lỗi xử lý: {str(e)}"