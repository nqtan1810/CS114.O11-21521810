
# SỐ HÓA TỦ SÁCH (DIGITALIZING BOOKSHELVES)

Đề tài nghiên cứu, phát triển hệ thống với chức năng nhận diện các vùng văn bản trên ảnh bìa sách, phân loại đâu là tên sách và  xuất ra tên sách tương ứng.


## Sinh viên thực hiện
Nhóm sinh viên trường Đại học Công nghệ Thông tin Đại học Quốc gia-TP.HCM :
- [Nguyễn Gia Bảo Ngọc – 21520366](https://github.com/ngbn111723)
- [Nguyễn Quốc Trường An – 21521810](https://github.com/nqtan1810)
- [Nguyễn Đức Tú – 21521612](https://github.com/Tund272)



## Giảng viên hướng dẫn
- PhGS.TS. Lê Đình Duy
- ThS. Phạm Nguyễn Trường An
## Mục lục
1. [ Mở đầu. ](#mở-đầu)
2. [ Công cụ và ngôn ngữ lập trình. ](#công-cụ-và-ngôn-ngữ-lập-trình)
3. [ Tổng quan. ](#tổng-quan)
- [ Mô tả bài toán ](#mô-tả-bài-toán)
- [ Dataset ](#dataset)
- [ Pipeline và Model ](#pipeline-và-model)
4. [ Chi tiết thực hiện. ](#chi-tiết-thực-hiện)
- [ Text Detection ](#text-detection)
- [ Binary Classification ](#binary-classification)
- [ Text Extraction ](#text-extraction)
5. [ Đánh giá và tổng kết. ](#đánh-giá-và-tổng-kết)
6. [ Tài liệu tham khảo. ](#tài-liệu-tham-khảo)


## Mở đầu
Ngày nay với sự phát triển của công nghệ thông tin và trí tuệ nhân tạo, việc ứng dụng Máy Học (Machine Learning) vào đời sống thực tế ngày càng trở nên phổ biến. Xuất phát từ thực tế là cá nhân hay tổ chức (thư viện, hiệu sách cũ,...) sở hữu số lượng sách lớn sẽ có nhu cầu lập danh sách quản lí sách thuộc sở hữu. Trong trường hợp lượng lớn sách cần lập danh sách, sẽ khiến cho việc nhập liệu thủ công trở nên mất nhiều công sức và thời gian. Nhìn thấy được điều này, đề tài nghiên cứu của nhóm với mục tiêu là phát triển một hệ thống bao gồm mô hình Máy học và các công cụ hỗ trợ khác, cho phép người dùng từ ảnh là trang bìa một cuốn sách nhận dạng được và xuất ra tên của quyển sách đó, hỗ trợ việc lập danh sách quản lý sách.

![Làm thế nào để lập một danh sách với hơn một triệu phần tử?](https://drive.google.com/uc?export=view&id=1pTW5GRaXK7S0HYJMJEtmDAJ4FQ52F7rS)



## Công cụ và ngôn ngữ lập trình

Project được cài đặt bằng ngôn ngữ Python và chạy trên Google Colab.
- Ngôn ngữ lập trình: Python
- Môi trường lập trình: Google Colab


## Tổng quan
### Mô tả bài toán
Đề tài nhóm thực hiện với mục tiêu là phát triển hệ thống bao gồm mô hình MMáy học chuyên biệt kết hợp với công cụ hỗ trợ cho phép người dùng đưa vào ảnh là trang bìa một cuốn sách, mô hình sẽ nhận dạng được và xuất ra tên cuốn sách đó, hỗ trợ việc lập danh sách quản lí sách.

#### Input
- Input: Ảnh bìa quyển sách

#### Output
- Output: Tên của quyển sách

#### Ngữ cảnh ứng dụng
Sản phẩm của đề tài có thể ứng dụng cho các tổ chức, cá nhân có nhu cầu lập danh sách quản lí sách, văn bản có bìa như sách,… Ví dụ thực tế: Các tiệm sách cũ, các quán cà phê sách, thư viện, tiệm sách cũ… có thể sở hữu hàng trăm thậm chí hàng nghìn tựa sách, có thể từ nhiều nguồn như mua lại, được cho tặng mà các trường hợp trên không có danh sách quản lí ngay từ ban đầu, việc đó thường gây khó khăn cho việc quản lí sách, bởi lẽ số lượng lớn sách dẫn đến việc nhập liệu thủ công tốn thời gian.

#### Lý do sử dụng mô hình máy học
Trên một bìa sách có thể bao gồm rất nhiều vùng văn bản, nhưng không phải tên sách, ví dụ như: Tên nhà xuất bản, tên tác giả hoặc cụm từ chỉ để trang trí hay quảng bá như: “Best Seller”, “New Edition”,... mà không có bất kì chương trình lập trình truyền thống cụ thể nào có thể phân loại được đâu là tựa sách từ các vùng văn bản được nhận diện trên ảnh bìa sách. Vì những lí do trên, hệ thống mà nhóm phát triển cần áp dụng một mô hình Máy học cụ thể là mô hình Binary Classification với mục đích chính là phân loại vùng văn bản chứa tựa sách với các vùng văn bản khác được nhận diện trên ảnh bìa sách.
### Tập dữ liệu
- Tập dữ liệu bao gồm 5046 phần tử là các đặc trưng được rút trích từ 700 ảnh bìa sách tiếng Việt hoặc sách tiếng Anh.
- Nguồn thu thập dữ liệu: từ cá nhân, Thư viện Đại học Công nghệ Thông tin ĐHQG-TP.HCM, Thư viện Trung tâm ĐHQG-TP.HCM, Thư viện Trung tâm ĐHQG-TP.HCM chi nhánh KTX khu B, các nguồn công khai khác... 
- Số lượng phần tử trong tập train và test sẽ lần lượt theo tỉ lệ 75% và 25%.
![Một phần của dataset](https://drive.google.com/uc?export=view&id=1K8kFzMt7Rq_WLAiOv93EkhTNxAUdgXPI)
### CÁC GIAI ĐOẠN TRONG HỆ THỐNG
Đề tài thực hiện của nhóm bao gồm một hệ thống Pipeline gồm 3 mô hình máy học ứng với 3 giai đoạn xử lí của dữ liệu: 
- Giai đoạn thứ nhất (Text Detection): có chức năng nhận diện ra vùng văn bản (hay vùng ảnh có chứa văn bản). Giai đoạn này được thực hiện bằng cách áp dụng công Pytesseract hỗ trợ việc nhận diện vùng văn bản trên ảnh bìa sách.
- Giai đoạn thứ hai (Classification): có chức năng phân loại vùng văn bản có chứa tên sách với các vùng văn bản khác được nhận diện từ ảnh bìa sách. Giai đoạn này được thực hiện bằng cách áp dụng mô hình máy học Binary Classification (đây là mô hình Máy học nhóm sẽ thu thập dữ liệu và huấn luyện).
- Giai đoạn thứ ba (Text Extraction): có chức năng là xuất vùng văn bản có trong ảnh. Sau khi đã biết được vùng nào là vùng chứa tên sách. Giai đoạn này sẽ sử dụng công cụ hỗ trợ là Easy OCR để xuất văn bản từ vùng ảnh có chứa văn bản được phân loại là tựa sách.
![Kết quả output của Pipeline](https://drive.google.com/uc?export=view&id=1g-AZ2dUmEhxeAcvGsMh5t4x3-3qN_LGo)


## Chi tiết thực hiện
Phần này sẽ đi sâu vào chi tiết những công đoạn mà nhóm đã nghiên cứu và thực hiện để cho ra kết quả cuối cùng, hình sau là Machine Learning Pipeline của nhóm:
![Machine Learning Pipeline](https://drive.google.com/uc?export=view&id=1f5EIJJpOaYcxq5MPhT5jsKaDmUSx_fIH)
### Giai đoạn thứ nhất (Text Detection):
Đây là bước nhận diện vùng văn bản có trên ảnh bìa sách, được hỗ trợ bởi công cụ Pytesseract.
![Khoang vùng văn bản](https://drive.google.com/uc?export=view&id=1GzStPu-1IpdEENrA4RzSkqtvSfVNcBVv)
### Giai đoạn thứ hai (Classification):
Để có được mô hình Binary Classification tốt nhất cho giai đoạn này của hệ thống, nhóm đã thực hiện huấn luyện 6 mô hình Binary Classification dựa trên 6 thuật toán khác nhau trên tập dữ liệu có được để chọn ra mô hình tốt nhất cho hệ thống. Cụ thể 6 thuật toán được áp dụng là:
- Logistic regression 
- Support Vector Machine 
- Decision Tree
- Random Forest 
- Naïve Bayes
- K-nearest Neighbor 
 
Các mô hình Binary Classification này đều đã được tích hợp sẵn trong thư viện ‘sklearn’ nhóm sẽ tiến hành sử dụng như sử dụng các đối tượng bình thường khác. Sau các bước huấn luyện, nhóm thực hiện sẽ chọn ra mô hình có thông F1-Score cao nhất để thực hiện cài đặt vào hệ thống.  
![Thông số các mô hình Binary Classification](https://drive.google.com/uc?export=view&id=1j8IHCbA3nrW4Ql9k36Be4n-uY8zPzw5E)

### Text Extraction
Sau khi đã biết được vùng văn  nào là vùng chứa tên sách, giai đoạn này sẽ sử dụng công cụ hỗ trợ là Easy OCR để xuất văn bản từ vùng văn bản được phân loại là tựa sách, cho ra kết quả cuối cùng.
![Text Extraction](https://drive.google.com/uc?export=view&id=1AK956UcxIrKf6zAlutw4_Lcm2v-5Rs2c)

## Đánh giá và tổng kết
Thực hiện đánh giá với tập test đầu vào là 70 ảnh bìa sách, kết quả thu được như sau:
- Kết quả đánh giá hiệu suất của giai đoạn thứ nhất (Text Detection): 66%
- Kết quả đánh giá hiệu suất của giai đoạn thứ hai (Classification): 81%
- Kết quả đánh giá hiệu suất của giai đoạn thứ ba (Text Extraction): 43.6%
- Kết quả đánh giá độ chính xác của toàn bộ hệ thống: 7.2%
## Tài liệu tham khảo
- [Text Detection and Extraction using OpenCV and OCR](https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/amp/?fbclid=IwAR158PXv1lY__2vw2dfHXbhorWqO-WnoYhJzgbU1vOckgnh-Hjjn5-jrz1Y)
- [Binary Classification](https://www.learndatasci.com/glossary/binary-classification/?fbclid=IwAR3QBvsHXBle5sFrntfTgwZWrWAIWAWtOYSfJxBjZ42DCkPuQ_RSO5zADoY)
- [Extraction from Book Cover Images](https://www.researchgate.net/publication/271130671_Title_Extraction_from_Book_Cover_Images_Using_Histogram_of_Oriented_Gradients_and_Color_Information?fbclid=IwAR1QwvQfhpLMCkLiPm6KAkUS-syQb0MDBDG9QuDznKQTP4EowBXQmWz5f-4)
- [VIETNAMESE TEXT EXTRACTION FROM BOOK COVERS](https://www.researchgate.net/publication/339359700_VIETNAMESE_TEXT_EXTRACTION_FROM_BOOK_COVERS?fbclid=IwAR0DPQxXAb1Fwq6w89Geon_mM7ILWA1ny90abGoQiRmU6iW4P9hRtrYQKBk)
- [Extract Title from the Image documents in python — Application of RLSA](https://vasista.medium.com/extract-title-from-the-image-documents-in-python-application-of-rlsa-58f91237901f)

