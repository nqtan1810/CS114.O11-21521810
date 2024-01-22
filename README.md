
# SỐ HÓA TỦ SÁCH (DIGITALIZING BOOKSHELVES)

Đề tài nghiên cứu phát triển hệ thống cho phép người dùng chụp bìa cuốn sách, và hệ thống sẽ tự động nhận dạng chữ trên đó, phân biệt đâu là tên sách và xuất ra tên sách tương ứng, hỗ trợ việc lập danh sách quản lí sách.


## Người thực hiện
Nhóm sinh viên trường Đại học Công nghệ Thông tin:
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
Ngày nay với sự phát triển của công nghệ thông tin và trí tuệ nhân tạo AI, việc ứng dụng Machine Learning vào đời sống thực tế hàng ngày càng trở nên dễ dàng và hiệu quả. Xuất phát từ nhu cầu thực tế việc cần quản lí một lượng lớn số lượng sách mà không có danh sách quản lí ngay từ đầu, việc lập danh sách quản lí là việc thiết thực và cần thiết. Bởi số lượng lớn sách cần lập danh sách khiến cho việc nhập liệu thủ công trở nên tốn rất nhiều công sức và thời gian. Nhìn thấy được điều này, đề tài nghiên cứu của nhóm với mục tiêu là phát triển một mô hình máy học cho phép người dùng đưa vào ảnh là trang bìa một cuốn sách nhận dạng được và xuất ra tên của quyển sách đó, hỗ trợ việc lập danh sách quản lý sách.

![Làm thế nào để lập một danh sách với hơn một triệu phần tử?](https://drive.google.com/uc?export=view&id=1pTW5GRaXK7S0HYJMJEtmDAJ4FQ52F7rS)



## Công cụ và ngôn ngữ lập trình

Project được cài đặt bằng ngôn ngữ Python và chạy trên Google Colab.
- Ngôn ngữ lập trình: Python
- Phần mềm lập trình: Google Colab


## Tổng quan
### Mô tả bài toán
Đề tài nhóm thực hiện với mục tiêu là phát triển hệ thống bao gồm mô hình máy học chuyên biệt kết hợp với công cụ hỗ trợ cho phép người dùng đưa vào ảnh là trang bìa một cuốn sách, mô hình sẽ nhận dạng được và xuất ra tên cuốn sách đó, hỗ trợ việc lập danh sách quản lí sách.

#### Input
- Input: Ảnh bìa quyển sách

#### Output
- Output: Tên của quyển sách

#### Ngữ cảnh ứng dụng
Sản phẩm của đề tài có thể ứng dụng cho các tổ chức, cá nhân có nhu cầu lập danh sách quản lí sách, văn bản có bìa như sách,… Ví dụ thực tế: Các tiệm sách cũ, các quán cà phê sách, thư viện, tiệm sách cũ… có thể sở hữu hàng trăm thậm chí hàng nghìn tựa sách, có thể từ nhiều nguồn như mua lại, được cho tặng mà các trường hợp trên không có danh sách quản lí ngay từ ban đầu, việc đó thường gây khó khăn cho việc quản lí sách, bởi lẽ số lượng lớn sách dẫn đến việc nhập liệu thủ công tốn thời gian.

#### Lý do sử dụng mô hình máy học
Trên một bìa sách có thể bao gồm rất nhiều kí tự chữ viết, nhưng không phải tên sách, bao gồm: Tên nhà xuất bản, tên tác giả hoặc cụm từ chỉ để trang trí hay marketing như “Best Seller”, “New Edition”… chưa kể đến chất lượng ảnh, góc chụp ảnh. Vì những lí do trên đề tài cần áp dụng mô hình máy học kết hợp các công cụ hỗ trợ như OpenCV, Easy OCR và Pytesseract để hỗ trợ việc cắt ảnh, nhận diện vùng có chữ (Text Detection) và tách chữ (Text Extraction), kết hợp với model máy học Binary Classification của nhóm phát triển thì từ input là ảnh bìa sẽ đưa ra được output cuối cùng là tên sách ở dạng chữ.
### Dataset
- Các phần tử trong bộ dữ liệu bao gồm ảnh của bìa sách. 
- Số lượng phần tử trong bộ dữ liệu bao gồm 5046 phần tử là các đặc trưng được rút trích từ 700 ảnh bìa sách tiếng Việt hoặc sách tiếng Anh.
- Nguồn thu thập dữ liệu: từ cá nhân, các thư viện các trường học, các nguồn công khai khác... 
- Các thao tác tiền xử lý (dự kiến): có thể chuyển sang định dạng trắng đen giúp giảm nhiễu là làm cho văn bản trên ảnh trở nên rõ ràng hơn.
- Số lượng phần tử trong tập train và test sẽ lần lượt theo tỉ lệ 75% và 25%.
- Trong phạm vi đề tài sẽ chỉ tập trung vào sách, các loại văn bản gần giống sách (báo và tạp chí không được đưa vào do trang bìa của báo và tạp chí có thể bao gồm nhiều tiêu đề, hình ảnh yêu cầu kĩ thuật cao hơn).
![Một phần của dataset](https://drive.google.com/uc?export=view&id=1K8kFzMt7Rq_WLAiOv93EkhTNxAUdgXPI)
### Pipeline và Model
Đề tài thực hiện của nhóm bao gồm một hệ thống Pipeline gồm 3 mô hình máy học ứng với 3 giai đoạn xử lí của dữ liệu: Text Detection, Binary Classification, Text Extraction. Trong đó, Binary Classification là phần nhiệm vụ chính của nhóm cần thực hiện training model.
- Giai đoạn thứ nhất (Text Detection): Ảnh bìa sách đưa vào được xử lí để nhận diện ra vùng chứa chữ và kí tự. Giai đoạn này được thực hiện bằng công cụ hỗ trợ Pytesseract để khoanh vùng chứ chữ và kí tự trên bìa sách.
- Giai đoạn thứ hai (Binary Classification): Nhóm sẽ thực hiện training model nhận diện các vùng có chữ này, đâu là vùng chứa tên sách. Giai đoạn này sử dụng mô hình máy học Binary Classification để thực hiện. Đây là giai đoạn mà nhóm sẽ tiến hành tự training model và cài đặt trong hệ thống Pipeline.
- Giai đoạn thứ ba (Text Extraction): Sau khi đã biết được vùng nào là vùng chứa tên sách, ở giai đoạn này sẽ sử dụng công cụ hỗ trợ là Easy OCR để tách chữ từ vùng nhận diện được trong ảnh thành văn bản.
![Kết quả output của Pipeline](https://drive.google.com/uc?export=view&id=1g-AZ2dUmEhxeAcvGsMh5t4x3-3qN_LGo)


## Chi tiết thực hiện
Phần này sẽ đi sâu vào chi tiết những công đoạn mà nhóm đã nghiên cứu và thực hiện để cho ra kết quả cuối cùng, hình sau là Machine Learning Pipeline của nhóm:
![Machine Learning Pipeline](https://drive.google.com/uc?export=view&id=1f5EIJJpOaYcxq5MPhT5jsKaDmUSx_fIH)
### Text Detection
Đây là bước nhận diện khoanh vùng chữ trên bìa sách sử dụng công cụ Pytesseract.
![Khoang vùng văn bản](https://drive.google.com/uc?export=view&id=1GzStPu-1IpdEENrA4RzSkqtvSfVNcBVv)
### Binary Classification
Để có được mô hình Binary Classification tốt nhất, nhóm đã thực hiện huấn luyện 6 mô hình Binary Classification dựa trên 6 thuật toán khác nhau trên tập dữ liệu được có được để chọn ra mô hình tốt nhất cho hệ thống. Cụ thể 6 thuật toán được áp dụng là:
- Logistic regression 
- Support Vector Machine 
- Decision Tree
- Random Forest 
- Naïve Bayes
- K-nearest Neighbor 
 
Các mô hình Binary Classification này đều đã được tích hợp sẵn trong thư viện ‘sklearn’ nhóm sẽ tiến hành sử dụng như sử dụng các đối tượng bình thường khác. Sau các bước training, nhóm thực hiện chọn ra model có F1-Score cao nhất để thực hiện cài đặt vào hệ thống Pipeline.  
![Thông số các mô hình Binary Classification](https://drive.google.com/uc?export=view&id=1j8IHCbA3nrW4Ql9k36Be4n-uY8zPzw5E)

### Text Extraction
Sau khi đã biết được vùng nào là vùng chứa tên sách, ở giai đoạn này sẽ sử dụng công cụ hỗ trợ là Easy OCR để tách chữ từ vùng nhận diện được trong ảnh thành văn bản, cho ra kết quả cuối cùng.
![Text Extraction](https://drive.google.com/uc?export=view&id=1AK956UcxIrKf6zAlutw4_Lcm2v-5Rs2c)

## Đánh giá và tổng kết
Thực hiện đánh giá với tập test đầu vào là 70 ảnh bìa sách, kết quả thu được như sau:
- Kết quả đánh giá accuracy của Text Detection: 66%
- Kết quả đánh giá accuracy của Binary Classification: 81%
- Kết quả đánh giá accuracy của Text Extraction: 43.6%
- Kết quả đánh giá Pipeline: 7.2%
## Tài liệu tham khảo
- [Text Detection and Extraction using OpenCV and OCR](https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/amp/?fbclid=IwAR158PXv1lY__2vw2dfHXbhorWqO-WnoYhJzgbU1vOckgnh-Hjjn5-jrz1Y)
- [Binary Classification](https://www.learndatasci.com/glossary/binary-classification/?fbclid=IwAR3QBvsHXBle5sFrntfTgwZWrWAIWAWtOYSfJxBjZ42DCkPuQ_RSO5zADoY)
- [Extraction from Book Cover Images](https://www.researchgate.net/publication/271130671_Title_Extraction_from_Book_Cover_Images_Using_Histogram_of_Oriented_Gradients_and_Color_Information?fbclid=IwAR1QwvQfhpLMCkLiPm6KAkUS-syQb0MDBDG9QuDznKQTP4EowBXQmWz5f-4)
- [VIETNAMESE TEXT EXTRACTION FROM BOOK COVERS](https://www.researchgate.net/publication/339359700_VIETNAMESE_TEXT_EXTRACTION_FROM_BOOK_COVERS?fbclid=IwAR0DPQxXAb1Fwq6w89Geon_mM7ILWA1ny90abGoQiRmU6iW4P9hRtrYQKBk)
- [Extract Title from the Image documents in python — Application of RLSA](https://vasista.medium.com/extract-title-from-the-image-documents-in-python-application-of-rlsa-58f91237901f)

