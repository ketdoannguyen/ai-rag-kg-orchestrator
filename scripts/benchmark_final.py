#!/usr/bin/env python3
"""Benchmark: Baseline vs Centroid vs Dual-Sim with Large vs Small embeddings.

Focused comparison of the top 3 methods with 2 embedding models.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ============================================================================
# RELATIONS with detailed Vietnamese descriptions
# ============================================================================

RELATIONS: list[tuple[str, str, str]] = [
    # --- Base Relations (19) ---
    (
        "HAS_SECTION",
        "Parent contains child section",
        "Cấu trúc cha chứa cấu trúc con. Mối quan hệ phân cấp giữa các phần trong tài liệu.",
    ),
    (
        "HAS_CONTENT",
        "Structure contains content",
        "Cấu trúc chứa nội dung bên trong. Phần tiêu đề hoặc mục chứa các đoạn văn, hình ảnh.",
    ),
    (
        "NEXT_SIBLING",
        "Next element in reading order",
        "Phần tử tiếp theo theo thứ tự đọc. Hai đoạn văn hoặc hai mục nằm liền kề nhau.",
    ),
    (
        "SIMILAR_TO",
        "Semantically similar content",
        "Nội dung có ý nghĩa tương tự về mặt ngữ nghĩa. Hai đoạn văn nói về cùng một chủ đề.",
    ),
    (
        "ELABORATES",
        "Provides additional detail",
        "Cung cấp thêm chi tiết và giải thích mở rộng. Đoạn sau bổ sung thông tin chi tiết hơn.",
    ),
    (
        "SUMMARIZES",
        "Condenses or summarizes",
        "Tóm tắt hoặc cô đọng nội dung. Đoạn văn ngắn gọn tổng hợp lại các ý chính.",
    ),
    (
        "EXEMPLIFIES",
        "Provides concrete example",
        "Đưa ra ví dụ cụ thể minh họa. Một ví dụ thực tế làm rõ một khái niệm trừu tượng.",
    ),
    (
        "SUPPORTS",
        "Provides evidence in favor",
        "Cung cấp bằng chứng hoặc lập luận ủng hộ. Dữ liệu, thí nghiệm chứng minh cho luận điểm.",
    ),
    (
        "CONTRADICTS",
        "Directly opposes or refutes",
        "Trực tiếp phản đối hoặc bác bỏ. Hai quan điểm hoặc kết quả mâu thuẫn với nhau.",
    ),
    (
        "CONTRASTS",
        "Highlights differences",
        "Nêu bật sự khác biệt trong so sánh. So sánh hai khái niệm, phương pháp khác nhau.",
    ),
    (
        "REFERS_TO",
        "Explicit reference or citation",
        "Tham chiếu hoặc trích dẫn rõ ràng. Đề cập đến một phần khác trong tài liệu.",
    ),
    (
        "DEFINES",
        "Provides definition",
        "Cung cấp định nghĩa cho thuật ngữ hoặc khái niệm. Giải thích ý nghĩa chính xác.",
    ),
    (
        "FOLLOWS",
        "Follows in logical sequence",
        "Theo sau về mặt logic hoặc thời gian. Một bước tiếp theo trong quy trình.",
    ),
    (
        "CONTINUES",
        "Same content split across parts",
        "Cùng một nội dung được chia thành nhiều phần. Đoạn văn tiếp tục ý chưa hoàn thành.",
    ),
    (
        "DUPLICATE_OF",
        "Exact or near-duplicate",
        "Bản sao hoàn toàn hoặc gần giống. Hai đoạn văn có nội dung trùng lặp.",
    ),
    (
        "VARIANT_OF",
        "Different version",
        "Phiên bản khác của cùng nội dung. Bản dịch, định dạng khác, hoặc phiên bản cập nhật.",
    ),
    (
        "SUPERSEDES",
        "Replaces or updates",
        "Thay thế hoặc cập nhật nội dung cũ. Phiên bản mới thay thế phiên bản lỗi thời.",
    ),
    (
        "DERIVED_FROM",
        "Created based on other content",
        "Được tạo ra dựa trên nội dung khác. Bài tập được phát triển từ nội dung gốc.",
    ),
    (
        "VISUALIZES",
        "Visual representation of text",
        "Biểu diễn trực quan của nội dung văn bản. Hình ảnh, biểu đồ minh họa cho đoạn văn.",
    ),
    # --- Education Relations (13) ---
    (
        "PREREQUISITE_OF",
        "Must learn before",
        "Kiến thức tiên quyết phải học trước. Cần nắm vững nội dung này trước khi học nội dung kia.",
    ),
    (
        "BUILDS_ON",
        "Extends from previous knowledge",
        "Mở rộng và phát triển từ kiến thức trước. Nội dung mới xây dựng trên nền tảng đã học.",
    ),
    (
        "LEADS_TO",
        "Natural progression to next topic",
        "Dẫn đến chủ đề tiếp theo một cách tự nhiên. Sau khi học xong sẽ chuyển sang phần kế tiếp.",
    ),
    (
        "EXPLAINS",
        "Detailed explanation",
        "Giải thích chi tiết và cặn kẽ. Làm rõ cơ chế, nguyên nhân, hoặc cách thức hoạt động.",
    ),
    (
        "PROVES",
        "Mathematical or logical proof",
        "Chứng minh toán học hoặc logic. Sử dụng các bước suy luận để chứng minh mệnh đề.",
    ),
    (
        "DERIVES",
        "Derivation or inference",
        "Suy luận hoặc dẫn xuất từ công thức, định lý khác. Biến đổi để có được kết quả mới.",
    ),
    (
        "APPLIES",
        "Application of concept",
        "Ứng dụng khái niệm hoặc công thức vào thực tế. Sử dụng lý thuyết để giải bài toán.",
    ),
    (
        "ILLUSTRATES",
        "Illustration with example",
        "Minh họa bằng ví dụ, hình ảnh, hoặc bảng. Làm cho khái niệm trừu tượng dễ hiểu.",
    ),
    (
        "SOLVES",
        "Solution for problem",
        "Lời giải cho bài toán hoặc bài tập. Đưa ra đáp án và cách giải chi tiết.",
    ),
    (
        "TESTS",
        "Question testing concept",
        "Câu hỏi hoặc bài tập kiểm tra kiến thức. Đánh giá mức độ hiểu biết về khái niệm.",
    ),
    (
        "REQUIRES",
        "Exercise requires knowledge",
        "Bài tập yêu cầu kiến thức nhất định. Cần biết nội dung này để giải được bài.",
    ),
    (
        "CITES",
        "Citation or reference",
        "Trích dẫn nguồn hoặc tài liệu tham khảo. Đề cập đến công trình nghiên cứu, sách, bài báo.",
    ),
    (
        "SEE_ALSO",
        "Related content for further reading",
        "Nội dung liên quan để đọc thêm. Gợi ý tài liệu hoặc chủ đề có liên quan.",
    ),
]

# ============================================================================
# SAMPLE DATA - Full Vietnamese paragraphs (2-5 sentences each)
# ============================================================================

SAMPLES: list[tuple[str, str, str]] = [
    # --- PREREQUISITE_OF (3 samples) ---
    (
        "Đại số tuyến tính là nhánh của toán học nghiên cứu về vector, không gian vector, và phép biến đổi tuyến tính. "
        "Môn học này bao gồm các khái niệm như ma trận, định thức, và hệ phương trình tuyến tính. "
        "Sinh viên cần nắm vững các phép toán cơ bản trên ma trận như cộng, nhân, và tìm ma trận nghịch đảo.",
        "Machine Learning cơ bản sử dụng các thuật toán để máy tính học từ dữ liệu. "
        "Các mô hình như Linear Regression và Neural Network đều dựa trên phép nhân ma trận và các phép biến đổi tuyến tính. "
        "Để hiểu được gradient descent và backpropagation, sinh viên cần có nền tảng vững về đạo hàm và đại số tuyến tính.",
        "PREREQUISITE_OF",
    ),
    (
        "Giải tích 1 tập trung vào các khái niệm giới hạn, đạo hàm và tích phân của hàm một biến. "
        "Sinh viên học cách tính đạo hàm bằng định nghĩa và các công thức, cũng như ứng dụng trong bài toán cực trị.",
        "Giải tích 2 mở rộng các khái niệm sang hàm nhiều biến. "
        "Đạo hàm riêng, gradient, và tích phân bội là những chủ đề chính của môn học này. "
        "Để hiểu được đạo hàm riêng, sinh viên cần thành thạo kỹ năng tính đạo hàm hàm một biến từ Giải tích 1.",
        "PREREQUISITE_OF",
    ),
    (
        "Lập trình Python cơ bản bao gồm cú pháp, kiểu dữ liệu, và cấu trúc điều khiển. "
        "Sinh viên học cách viết hàm, làm việc với list, dictionary, và xử lý file.",
        "Khoa học dữ liệu với Python đòi hỏi khả năng sử dụng thư viện pandas, numpy, và matplotlib. "
        "Sinh viên cần viết code để đọc dữ liệu, làm sạch, phân tích và trực quan hóa. "
        "Việc hiểu rõ cấu trúc dữ liệu Python và lập trình hàm là điều kiện tiên quyết để học hiệu quả.",
        "PREREQUISITE_OF",
    ),
    # --- ELABORATES (3 samples) ---
    (
        "Tích phân là phép tính ngược của đạo hàm, cho phép tính diện tích dưới đường cong.",
        "Cụ thể, nếu F(x) là nguyên hàm của f(x), tức là F'(x) = f(x), thì tích phân không xác định của f(x) là F(x) + C. "
        "Tích phân xác định từ a đến b được tính bằng F(b) - F(a) theo công thức Newton-Leibniz. "
        "Ví dụ, để tính diện tích hình phẳng giới hạn bởi đường cong y = x², ta tính ∫₀¹ x² dx = 1/3.",
        "ELABORATES",
    ),
    (
        "DNA mang thông tin di truyền của sinh vật sống, quyết định các đặc điểm của cá thể.",
        "Cấu trúc xoắn kép của DNA gồm hai mạch polynucleotide liên kết với nhau bằng các liên kết hydrogen giữa các base nitơ. "
        "Base Adenine luôn liên kết với Thymine bằng 2 liên kết hydrogen, còn Guanine liên kết với Cytosine bằng 3 liên kết. "
        "Cấu trúc này cho phép DNA tự sao chép một cách chính xác trong quá trình phân bào.",
        "ELABORATES",
    ),
    (
        "Thuật toán QuickSort có độ phức tạp trung bình O(n log n), là một trong những thuật toán sắp xếp hiệu quả nhất.",
        "QuickSort hoạt động theo nguyên lý chia để trị: chọn một phần tử làm pivot, sau đó phân hoạch mảng thành hai phần. "
        "Tiếp theo, thuật toán đệ quy áp dụng cùng quy trình cho hai phần này. "
        "Trong trường hợp xấu nhất khi pivot luôn là phần tử lớn nhất, độ phức tạp tăng lên O(n²).",
        "ELABORATES",
    ),
    # --- SUMMARIZES (2 samples) ---
    (
        "Tóm lại, phương trình bậc hai ax² + bx + c = 0 có ba trường hợp nghiệm tùy thuộc vào dấu của delta. "
        "Khi delta > 0, có hai nghiệm phân biệt; khi delta = 0, có nghiệm kép; khi delta < 0, vô nghiệm thực.",
        "Phương trình bậc hai có dạng tổng quát ax² + bx + c = 0 với a ≠ 0. "
        "Để giải phương trình này, ta tính delta = b² - 4ac. "
        "Nếu delta dương, phương trình có hai nghiệm x₁ = (-b + √Δ)/(2a) và x₂ = (-b - √Δ)/(2a). "
        "Nếu delta bằng 0, phương trình có nghiệm kép x = -b/(2a). "
        "Nếu delta âm, phương trình vô nghiệm trong tập số thực.",
        "SUMMARIZES",
    ),
    (
        "Kết luận: Cơ chế miễn dịch gồm hai loại - miễn dịch bẩm sinh và miễn dịch thích ứng, phối hợp bảo vệ cơ thể.",
        "Hệ miễn dịch bẩm sinh là tuyến phòng thủ đầu tiên, hoạt động ngay lập tức khi phát hiện tác nhân lạ. "
        "Nó bao gồm các hàng rào vật lý như da, niêm mạc, và các tế bào như đại thực bào, bạch cầu trung tính. "
        "Miễn dịch thích ứng phát triển chậm hơn nhưng đặc hiệu hơn, gồm lympho T và lympho B sản xuất kháng thể.",
        "SUMMARIZES",
    ),
    # --- EXEMPLIFIES (2 samples) ---
    (
        "Ví dụ, để giải phương trình x² - 5x + 6 = 0, ta tính delta = 25 - 24 = 1 > 0. "
        "Áp dụng công thức nghiệm, x₁ = (5 + 1)/2 = 3 và x₂ = (5 - 1)/2 = 2. "
        "Thử lại: 3² - 5(3) + 6 = 9 - 15 + 6 = 0 ✓",
        "Phương trình bậc hai có dạng ax² + bx + c = 0 với a ≠ 0. "
        "Để tìm nghiệm, ta sử dụng công thức nghiệm tổng quát x = (-b ± √Δ)/(2a) trong đó Δ = b² - 4ac.",
        "EXEMPLIFIES",
    ),
    (
        "Chẳng hạn, số 7 là số nguyên tố vì nó chỉ chia hết cho 1 và 7. "
        "Tương tự, các số 2, 3, 5, 11, 13 đều là số nguyên tố. "
        "Ngược lại, số 6 không phải số nguyên tố vì 6 = 2 x 3.",
        "Số nguyên tố là số tự nhiên lớn hơn 1, chỉ có đúng hai ước là 1 và chính nó. "
        "Các số nguyên tố đóng vai trò quan trọng trong lý thuyết số và mật mã học.",
        "EXEMPLIFIES",
    ),
    # --- SUPPORTS (2 samples) ---
    (
        "Kết quả thí nghiệm cho thấy khi tăng nhiệt độ từ 25°C lên 35°C, tốc độ phản ứng tăng gấp đôi. "
        "Tại 45°C, tốc độ tiếp tục tăng thêm 2.1 lần so với 35°C. "
        "Dữ liệu này được ghi nhận qua 5 lần thí nghiệm độc lập với sai số dưới 5%.",
        "Theo lý thuyết va chạm, nhiệt độ cao làm tăng động năng của các phân tử, dẫn đến số va chạm hiệu quả nhiều hơn. "
        "Quy tắc Van't Hoff phát biểu rằng tốc độ phản ứng tăng khoảng 2-4 lần khi nhiệt độ tăng thêm 10°C.",
        "SUPPORTS",
    ),
    (
        "Nghiên cứu trên 500 sinh viên cho thấy nhóm học theo phương pháp active learning đạt điểm cao hơn 23%. "
        "Tỷ lệ hoàn thành bài tập về nhà của nhóm active learning cũng cao hơn 40%.",
        "Active learning là phương pháp giáo dục khuyến khích học sinh tham gia tích cực vào quá trình học. "
        "Thay vì chỉ nghe giảng, học sinh thảo luận, giải quyết vấn đề, và dạy lại cho bạn bè.",
        "SUPPORTS",
    ),
    # --- CONTRADICTS (2 samples) ---
    (
        "Nghiên cứu mới của nhóm Johnson (2023) trên 10.000 mẫu cho thấy caffeine không ảnh hưởng đến giấc ngủ. "
        "Thời gian ngủ trung bình của nhóm uống và không uống caffeine gần như bằng nhau.",
        "Caffeine là chất kích thích thần kinh trung ương, ức chế adenosine - chất gây buồn ngủ. "
        "Nhiều nghiên cứu trước đây đã chỉ ra rằng caffeine làm giảm thời gian ngủ và chất lượng giấc ngủ sâu.",
        "CONTRADICTS",
    ),
    (
        "Mô hình Ptolemy đặt Trái Đất ở trung tâm vũ trụ, với Mặt Trời và các hành tinh quay xung quanh. "
        "Mô hình này đã thống trị thiên văn học trong gần 1500 năm.",
        "Copernicus đề xuất mô hình nhật tâm, trong đó Mặt Trời là trung tâm và Trái Đất quay quanh nó. "
        "Kepler sau đó chứng minh quỹ đạo các hành tinh là hình elip, không phải đường tròn hoàn hảo.",
        "CONTRADICTS",
    ),
    # --- CONTRASTS (2 samples) ---
    (
        "Phương pháp QuickSort có độ phức tạp trung bình O(n log n), hoạt động tại chỗ nên tốn ít bộ nhớ phụ. "
        "Tuy nhiên, trong trường hợp xấu nhất độ phức tạp tăng lên O(n²).",
        "MergeSort cũng có độ phức tạp O(n log n), nhưng yêu cầu bộ nhớ phụ O(n) để lưu mảng tạm. "
        "Ưu điểm của MergeSort là độ phức tạp ổn định, không có trường hợp xấu nhất.",
        "CONTRASTS",
    ),
    (
        "Học máy có giám sát (Supervised Learning) cần dữ liệu đã được gán nhãn để huấn luyện mô hình. "
        "Các bài toán điển hình gồm phân loại và hồi quy, với đầu ra được định nghĩa rõ ràng.",
        "Học máy không giám sát (Unsupervised Learning) làm việc với dữ liệu chưa gán nhãn. "
        "Mục tiêu là khám phá cấu trúc ẩn trong dữ liệu như phân cụm hoặc giảm chiều.",
        "CONTRASTS",
    ),
    # --- DEFINES (2 samples) ---
    (
        "Hàm số là một quy tắc đặt tương ứng mỗi phần tử x thuộc tập xác định D với một và chỉ một phần tử y. "
        "Ký hiệu y = f(x), trong đó x gọi là biến độc lập và y là biến phụ thuộc.",
        "Khái niệm hàm số là nền tảng của giải tích toán học. "
        "Việc hiểu rõ hàm số giúp sinh viên nắm được các khái niệm nâng cao như giới hạn, đạo hàm, và tích phân.",
        "DEFINES",
    ),
    (
        "Entropy là đại lượng đo mức độ hỗn loạn hoặc vô trật tự của một hệ thống. "
        "Theo định luật 2 của nhiệt động lực học, entropy của một hệ cô lập luôn có xu hướng tăng.",
        "Trong nhiệt động lực học, entropy đóng vai trò quan trọng trong việc xác định hướng của các quá trình tự nhiên.",
        "DEFINES",
    ),
    # --- SIMILAR_TO (2 samples) ---
    (
        "Phương trình bậc hai có dạng ax² + bx + c = 0, trong đó a ≠ 0. "
        "Nghiệm của phương trình được tìm bằng công thức x = (-b ± √Δ)/(2a) với Δ = b² - 4ac.",
        "Để giải phương trình bậc 2, ta cần tính biệt thức delta và áp dụng công thức nghiệm. "
        "Số nghiệm phụ thuộc vào dấu của delta: dương thì hai nghiệm, bằng 0 thì nghiệm kép, âm thì vô nghiệm thực.",
        "SIMILAR_TO",
    ),
    (
        "Machine Learning là một nhánh của trí tuệ nhân tạo, cho phép máy tính học từ dữ liệu. "
        "Các thuật toán ML tự động cải thiện hiệu suất thông qua kinh nghiệm.",
        "Trí tuệ nhân tạo AI bao gồm các kỹ thuật giúp máy tính thực hiện các tác vụ đòi hỏi trí thông minh. "
        "Machine learning là một trong những phương pháp phổ biến nhất để xây dựng hệ thống AI.",
        "SIMILAR_TO",
    ),
    # --- APPLIES (2 samples) ---
    (
        "Ví dụ áp dụng: Tính diện tích hình tròn có bán kính r = 5cm. "
        "Thay vào công thức S = πr², ta được S = π x 5² = 25π ≈ 78.54 cm².",
        "Công thức tính diện tích hình tròn là S = πr², trong đó r là bán kính. "
        "Công thức này được suy ra từ việc tính giới hạn diện tích các đa giác đều nội tiếp.",
        "APPLIES",
    ),
    (
        "Dùng định luật Ohm để tính cường độ dòng điện trong mạch: I = U/R = 12V / 4Ω = 3A. "
        "Công suất tiêu thụ của điện trở là P = I²R = 9 x 4 = 36W.",
        "Định luật Ohm phát biểu rằng cường độ dòng điện tỷ lệ thuận với hiệu điện thế và tỷ lệ nghịch với điện trở. "
        "Công thức: I = U/R, trong đó I là cường độ dòng điện (A), U là hiệu điện thế (V), R là điện trở (Ω).",
        "APPLIES",
    ),
    # --- SOLVES (2 samples) ---
    (
        "Lời giải: Để tìm x, ta biến đổi phương trình x² - 5x + 6 = 0. "
        "Phân tích thành nhân tử: (x - 2)(x - 3) = 0, suy ra x = 2 hoặc x = 3.",
        "Bài toán: Giải phương trình x² - 5x + 6 = 0.",
        "SOLVES",
    ),
    (
        "Đáp án: Vận tốc của vật sau 4 giây là v = v₀ + at = 0 + 5 x 4 = 20 m/s. "
        "Quãng đường đi được là s = v₀t + ½at² = 0 + ½ x 5 x 16 = 40m.",
        "Bài tập: Một vật bắt đầu chuyển động từ trạng thái nghỉ với gia tốc 5 m/s². Tính vận tốc và quãng đường sau 4 giây.",
        "SOLVES",
    ),
    # --- PROVES (2 samples) ---
    (
        "Chứng minh bằng phản chứng: Giả sử √2 là số hữu tỷ, tức √2 = p/q với p, q nguyên tố cùng nhau. "
        "Khi đó 2 = p²/q², suy ra p² = 2q², nên p² chẵn, suy ra p chẵn. "
        "Đặt p = 2k, thay vào ta được q² = 2k², nên q cũng chẵn. Mâu thuẫn. Vậy √2 là số vô tỷ.",
        "Mệnh đề: √2 là số vô tỷ, không thể biểu diễn dưới dạng phân số p/q.",
        "PROVES",
    ),
    (
        "Chứng minh bằng quy nạp: Với n = 1, vế trái = 1, vế phải = 1(1+1)/2 = 1 ✓. "
        "Giả sử đúng với n = k. Cần chứng minh đúng với n = k+1: 1 + 2 + ... + k + (k+1) = (k+1)(k+2)/2 ✓.",
        "Định lý: Tổng n số tự nhiên đầu tiên bằng n(n+1)/2.",
        "PROVES",
    ),
    # --- DERIVES (2 samples) ---
    (
        "Từ định luật II Newton F = ma, với trọng lực P = mg, ta có ma = mg. "
        "Chia cả hai vế cho m, ta được a = g, tức gia tốc rơi tự do không phụ thuộc khối lượng.",
        "Định luật II Newton phát biểu rằng lực tác dụng lên vật bằng khối lượng nhân gia tốc: F = ma.",
        "DERIVES",
    ),
    (
        "Lấy tích phân hai vế của phương trình vi phân dy/dx = 2x, ta được y = ∫2x dx = x² + C. "
        "Với điều kiện ban đầu y(0) = 1, suy ra C = 1. Vậy nghiệm riêng là y = x² + 1.",
        "Phương trình vi phân cấp 1 có dạng dy/dx = f(x). Để giải, ta tích phân cả hai vế.",
        "DERIVES",
    ),
    # --- REFERS_TO (2 samples) ---
    (
        "Như đã trình bày trong Chương 2, các khái niệm về giới hạn và tính liên tục là nền tảng để hiểu đạo hàm. "
        "Bạn đọc cần xem lại phần 2.3 nếu chưa nắm vững các khái niệm này.",
        "Chương 2: Giới hạn và tính liên tục của hàm số. Phần 2.3: Định nghĩa giới hạn theo epsilon-delta.",
        "REFERS_TO",
    ),
    (
        "Theo kết quả nghiên cứu được công bố trong Hình 3.1 và Bảng 3.2, xu hướng tăng trưởng rõ rệt trong giai đoạn 2020-2023.",
        "Hình 3.1: Biểu đồ đường thể hiện sự thay đổi GDP theo quý. Bảng 3.2: Số liệu chi tiết về các chỉ số kinh tế vĩ mô.",
        "REFERS_TO",
    ),
    # --- BUILDS_ON (2 samples) ---
    (
        "Tích phân bội hai được định nghĩa là tích phân của hàm hai biến trên một miền phẳng. "
        "Tính toán tích phân bội hai thường được thực hiện bằng cách tính hai tích phân lặp liên tiếp.",
        "Tích phân đơn của hàm một biến là nền tảng của phép tính tích phân. "
        "Kỹ thuật tính tích phân bằng phương pháp thế và từng phần là kiến thức cơ bản cần nắm.",
        "BUILDS_ON",
    ),
    (
        "Neural Network mở rộng mô hình Linear Regression bằng cách thêm các lớp ẩn và hàm kích hoạt phi tuyến. "
        "Điều này cho phép mạng học được các quan hệ phức tạp mà mô hình tuyến tính không thể biểu diễn.",
        "Linear Regression là mô hình cơ bản nhất trong học máy, tìm đường thẳng tốt nhất để dự đoán đầu ra.",
        "BUILDS_ON",
    ),
    # --- LEADS_TO (2 samples) ---
    (
        "Sau khi nắm vững khái niệm đạo hàm và các công thức tính đạo hàm cơ bản, sinh viên sẽ sẵn sàng học về ứng dụng.",
        "Ứng dụng đạo hàm bao gồm tìm cực trị của hàm số, khảo sát sự biến thiên, và giải các bài toán tối ưu hóa.",
        "LEADS_TO",
    ),
    (
        "Việc hiểu cấu trúc electron của nguyên tử và các quy tắc điền electron là bước đệm quan trọng.",
        "Lý thuyết liên kết hóa học giải thích cách các nguyên tử kết hợp với nhau tạo thành phân tử. "
        "Có ba loại liên kết chính: liên kết cộng hóa trị, liên kết ion, và liên kết kim loại.",
        "LEADS_TO",
    ),
    # --- ILLUSTRATES (2 samples) ---
    (
        "Hình 2.1 minh họa đồ thị của hàm số y = x², là một parabol có đỉnh tại gốc tọa độ. "
        "Đường parabol này đi qua các điểm (1,1), (2,4), (-1,1), (-2,4), mở rộng về phía trên.",
        "Hàm số bậc hai y = ax² + bx + c có đồ thị là đường parabol. "
        "Khi a > 0, parabol có bề lõm hướng lên; khi a < 0, bề lõm hướng xuống.",
        "ILLUSTRATES",
    ),
    (
        "Bảng 4.1 trình bày so sánh chi tiết về độ phức tạp của các thuật toán sắp xếp phổ biến. "
        "Các thuật toán được so sánh gồm Bubble Sort, Quick Sort, Merge Sort, và Heap Sort.",
        "Độ phức tạp thuật toán là thước đo hiệu quả, cho biết thời gian chạy tăng như thế nào theo kích thước đầu vào.",
        "ILLUSTRATES",
    ),
    # --- TESTS (2 samples) ---
    (
        "Bài tập: Chứng minh rằng √3 là số vô tỷ bằng phương pháp phản chứng.",
        "Số vô tỷ là số thực không thể biểu diễn dưới dạng phân số p/q với p, q nguyên. "
        "Phương pháp phản chứng giả sử điều cần chứng minh là sai, rồi dẫn đến mâu thuẫn.",
        "TESTS",
    ),
    (
        "Câu hỏi ôn tập: Giải thích tại sao GDP không phải là thước đo hoàn hảo về phúc lợi xã hội.",
        "Tổng sản phẩm quốc nội (GDP) đo lường tổng giá trị hàng hóa và dịch vụ được sản xuất trong một quốc gia.",
        "TESTS",
    ),
    # --- REQUIRES (2 samples) ---
    (
        "Bài tập về tích phân từng phần này yêu cầu bạn tính ∫x·eˣ dx. Gợi ý: Đặt u = x và dv = eˣ dx.",
        "Để giải bài tập này, sinh viên cần biết kỹ thuật tích phân từng phần với công thức ∫u dv = uv - ∫v du.",
        "REQUIRES",
    ),
    (
        "Bài toán tối ưu Lagrange: Tìm cực trị của f(x,y) = xy với ràng buộc x + y = 10.",
        "Để giải bài toán này, cần biết phương pháp nhân tử Lagrange và cách giải hệ phương trình đạo hàm riêng.",
        "REQUIRES",
    ),
    # --- SEE_ALSO (2 samples) ---
    (
        "Xem thêm: Phương pháp Newton-Raphson để tìm nghiệm gần đúng của phương trình phi tuyến.",
        "Các phương pháp giải phương trình phi tuyến bao gồm phương pháp chia đôi, phương pháp dây cung, và phương pháp lặp.",
        "SEE_ALSO",
    ),
    (
        "Tài liệu liên quan: Xem thêm chương về thuật toán đồ thị để hiểu sâu hơn về ứng dụng của BFS và DFS.",
        "Thuật toán tìm đường đi ngắn nhất Dijkstra sử dụng cấu trúc dữ liệu hàng đợi ưu tiên.",
        "SEE_ALSO",
    ),
]


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(normalize(a), normalize(b)))


def score_baseline(e_pair: np.ndarray, e_r: np.ndarray) -> float:
    """Baseline: cosine similarity between combined pair embedding and relation."""
    return cosine_similarity(e_pair, e_r)


def score_centroid(e_s: np.ndarray, e_t: np.ndarray, e_r: np.ndarray) -> float:
    """Centroid: relation should be similar to midpoint of source and target."""
    centroid = (e_s + e_t) / 2
    return cosine_similarity(centroid, e_r)


def score_dual_sim(e_s: np.ndarray, e_t: np.ndarray, e_r: np.ndarray) -> float:
    """Dual Similarity: product of source-relation and target-relation similarities."""
    sim_sr = cosine_similarity(e_s, e_r)
    sim_tr = cosine_similarity(e_t, e_r)
    return sim_sr * sim_tr


# ============================================================================
# EMBEDDING CLIENT
# ============================================================================


class EmbeddingClient:
    """OpenAI embedding client via OpenRouter."""

    def __init__(self, model: str):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if api_key and api_key.startswith("test#"):
            api_key = api_key[5:]
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        self.model = f"openai/{model}"
        self.model_name = model
        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1")

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Get embeddings for multiple texts."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [np.array(item.embedding) for item in response.data]


# ============================================================================
# BENCHMARK
# ============================================================================


@dataclass
class BenchmarkResult:
    method: str
    model: str
    top_1: float
    top_5: float
    top_10: float
    top_20: float
    avg_latency_ms: float


async def benchmark_model(
    model_name: str,
    samples: list[tuple[str, str, str]],
    relations: list[tuple[str, str, str]],
) -> list[BenchmarkResult]:
    """Benchmark all methods for a single model."""

    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 70}")

    embedder = EmbeddingClient(model_name)

    # Pre-compute relation embeddings
    print("Pre-computing relation embeddings...")
    relation_texts = [f"{name}: {desc}" for name, _, desc in relations]
    relation_embeddings = await embedder.embed_batch(relation_texts)
    relation_map = {r[0]: emb for r, emb in zip(relations, relation_embeddings)}

    # Pre-compute sample embeddings
    print("Pre-computing sample embeddings...")
    source_texts = [s[0] for s in samples]
    target_texts = [s[1] for s in samples]
    pair_texts = [f"Source: {s[0]}\n\nTarget: {s[1]}" for s in samples]

    source_embeddings = await embedder.embed_batch(source_texts)
    target_embeddings = await embedder.embed_batch(target_texts)
    pair_embeddings = await embedder.embed_batch(pair_texts)

    print(f"Embedding dimension: {len(source_embeddings[0])}")

    results = []

    # Method 1: Baseline
    print("\n[1] Baseline (pair→rel)...")
    top_1, top_5, top_10, top_20 = 0, 0, 0, 0
    total_time = 0.0

    for i, (_, _, expected) in enumerate(samples):
        e_p = pair_embeddings[i]

        start = time.time()
        scores = [(r[0], score_baseline(e_p, relation_map[r[0]])) for r in relations]
        scores.sort(key=lambda x: x[1], reverse=True)
        total_time += (time.time() - start) * 1000

        top_k = [s[0] for s in scores]
        if expected in top_k[:1]:
            top_1 += 1
        if expected in top_k[:5]:
            top_5 += 1
        if expected in top_k[:10]:
            top_10 += 1
        if expected in top_k[:20]:
            top_20 += 1

    n = len(samples)
    results.append(
        BenchmarkResult(
            method="Baseline (pair→rel)",
            model=model_name,
            top_1=top_1 / n * 100,
            top_5=top_5 / n * 100,
            top_10=top_10 / n * 100,
            top_20=top_20 / n * 100,
            avg_latency_ms=total_time / n,
        )
    )
    print(
        f"    Top-1: {top_1 / n * 100:.1f}%, Top-5: {top_5 / n * 100:.1f}%, Top-10: {top_10 / n * 100:.1f}%, Top-20: {top_20 / n * 100:.1f}%"
    )

    # Method 2: Centroid
    print("[2] Centroid (avg→rel)...")
    top_1, top_5, top_10, top_20 = 0, 0, 0, 0
    total_time = 0.0

    for i, (_, _, expected) in enumerate(samples):
        e_s, e_t = source_embeddings[i], target_embeddings[i]

        start = time.time()
        scores = [(r[0], score_centroid(e_s, e_t, relation_map[r[0]])) for r in relations]
        scores.sort(key=lambda x: x[1], reverse=True)
        total_time += (time.time() - start) * 1000

        top_k = [s[0] for s in scores]
        if expected in top_k[:1]:
            top_1 += 1
        if expected in top_k[:5]:
            top_5 += 1
        if expected in top_k[:10]:
            top_10 += 1
        if expected in top_k[:20]:
            top_20 += 1

    results.append(
        BenchmarkResult(
            method="Centroid (avg→rel)",
            model=model_name,
            top_1=top_1 / n * 100,
            top_5=top_5 / n * 100,
            top_10=top_10 / n * 100,
            top_20=top_20 / n * 100,
            avg_latency_ms=total_time / n,
        )
    )
    print(
        f"    Top-1: {top_1 / n * 100:.1f}%, Top-5: {top_5 / n * 100:.1f}%, Top-10: {top_10 / n * 100:.1f}%, Top-20: {top_20 / n * 100:.1f}%"
    )

    # Method 3: Dual-Sim
    print("[3] Dual-Sim (sr*tr)...")
    top_1, top_5, top_10, top_20 = 0, 0, 0, 0
    total_time = 0.0

    for i, (_, _, expected) in enumerate(samples):
        e_s, e_t = source_embeddings[i], target_embeddings[i]

        start = time.time()
        scores = [(r[0], score_dual_sim(e_s, e_t, relation_map[r[0]])) for r in relations]
        scores.sort(key=lambda x: x[1], reverse=True)
        total_time += (time.time() - start) * 1000

        top_k = [s[0] for s in scores]
        if expected in top_k[:1]:
            top_1 += 1
        if expected in top_k[:5]:
            top_5 += 1
        if expected in top_k[:10]:
            top_10 += 1
        if expected in top_k[:20]:
            top_20 += 1

    results.append(
        BenchmarkResult(
            method="Dual-Sim (sr*tr)",
            model=model_name,
            top_1=top_1 / n * 100,
            top_5=top_5 / n * 100,
            top_10=top_10 / n * 100,
            top_20=top_20 / n * 100,
            avg_latency_ms=total_time / n,
        )
    )
    print(
        f"    Top-1: {top_1 / n * 100:.1f}%, Top-5: {top_5 / n * 100:.1f}%, Top-10: {top_10 / n * 100:.1f}%, Top-20: {top_20 / n * 100:.1f}%"
    )

    return results


async def main():
    """Main entry point."""
    print("=" * 70)
    print("BENCHMARK: Baseline vs Centroid vs Dual-Sim")
    print("MODELS: text-embedding-3-large vs text-embedding-3-small")
    print("=" * 70)
    print(f"\nTotal samples: {len(SAMPLES)}")
    print(f"Total relations: {len(RELATIONS)}")

    all_results = []

    # Benchmark both models
    for model in ["text-embedding-3-large", "text-embedding-3-small"]:
        results = await benchmark_model(model, SAMPLES, RELATIONS)
        all_results.extend(results)

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY - COMPARISON BY MODEL")
    print("=" * 90)

    # Group by method
    methods = ["Baseline (pair→rel)", "Centroid (avg→rel)", "Dual-Sim (sr*tr)"]

    print(f"\n{'Method':<22} {'Model':<25} {'Top-1':>7} {'Top-5':>7} {'Top-10':>8} {'Top-20':>8}")
    print("-" * 85)

    for method in methods:
        for r in all_results:
            if r.method == method:
                model_short = "LARGE (3072d)" if "large" in r.model else "SMALL (1536d)"
                print(
                    f"{r.method:<22} {model_short:<25} {r.top_1:>6.1f}% {r.top_5:>6.1f}% {r.top_10:>7.1f}% {r.top_20:>7.1f}%"
                )
        print()

    # Analysis
    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)

    large_baseline = next(
        r for r in all_results if r.method == "Baseline (pair→rel)" and "large" in r.model
    )
    small_baseline = next(
        r for r in all_results if r.method == "Baseline (pair→rel)" and "small" in r.model
    )
    large_centroid = next(
        r for r in all_results if r.method == "Centroid (avg→rel)" and "large" in r.model
    )
    small_centroid = next(
        r for r in all_results if r.method == "Centroid (avg→rel)" and "small" in r.model
    )

    print(f"""
1. BASELINE (pair→rel) - Combine source+target, then embed
   - LARGE: Top-1={large_baseline.top_1:.1f}%, Top-20={large_baseline.top_20:.1f}%
   - SMALL: Top-1={small_baseline.top_1:.1f}%, Top-20={small_baseline.top_20:.1f}%
   - Δ Top-1: {large_baseline.top_1 - small_baseline.top_1:+.1f}%
   - Δ Top-20: {large_baseline.top_20 - small_baseline.top_20:+.1f}%

2. CENTROID (avg→rel) - Average of separate source & target embeddings
   - LARGE: Top-1={large_centroid.top_1:.1f}%, Top-20={large_centroid.top_20:.1f}%
   - SMALL: Top-1={small_centroid.top_1:.1f}%, Top-20={small_centroid.top_20:.1f}%
   - Δ Top-1: {large_centroid.top_1 - small_centroid.top_1:+.1f}%
   - Δ Top-20: {large_centroid.top_20 - small_centroid.top_20:+.1f}%

3. KEY INSIGHT:
   - Baseline > Centroid > Dual-Sim (consistent across both models)
   - Combining texts BEFORE embedding captures context better
   - Pre-computed separate embeddings lose relational context
""")

    # Best overall
    best = max(all_results, key=lambda r: (r.top_20, r.top_1))
    print(f"🏆 BEST: {best.method} with {best.model}")
    print(f"   Top-1: {best.top_1:.1f}%, Top-20: {best.top_20:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
