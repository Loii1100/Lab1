# Lab1
from PIL import Image
import numpy as np
import matplotlib.pylab as plt # Thêm để sử dụng plt.show() nếu cần xem ảnh

img = Image.open('bird.png')
img.show() # Lệnh này sẽ mở ảnh bằng trình xem ảnh mặc định của hệ thống.
# Nếu bạn muốn xem ảnh trong cửa sổ matplotlib:
# plt.imshow(img)
# plt.show()
import numpy as np
import imageio.v2 as iio
import matplotlib.pylab as plt

print("--- 2.2 Tải ảnh cơ bản và chuyển ảnh xám ---")
data = iio.imread('bird.png')
plt.imshow(data)
plt.title("Ảnh gốc")
plt.show()

# Chuyển đổi sang ảnh xám và hiển thị
data_gray = iio.imread('bird.png', as_gray=True)
plt.imshow(data_gray, cmap='gray')
plt.title("Ảnh xám")
plt.show()

# Ví dụ: Ảnh xám và giảm độ sâu bit (4 bit)
print("--- Grayscale và giảm độ sâu bit ---")
data_uint8 = iio.imread('bird.png', as_gray=True).astype(np.uint8)
cl = data_uint8 & 0xF0 # Giữ lại 4 bit quan trọng nhất
iio.imwrite('birdF0.png', cl)
print("Đã lưu ảnh 'birdF0.png' với độ sâu bit giảm.")
tmp = iio.imread('birdF0.png')
plt.imshow(tmp, cmap='gray')
plt.title("Ảnh xám 4 bit")
plt.show()
import numpy as np
import imageio.v2 as iio
import matplotlib.pylab as plt

print("--- 2.4 Màu sắc với hệ RGB (Hiển thị kênh xanh lá và xanh lam) ---")
data = iio.imread('bird.png') # Mặc định là RGB (Red, Green, Blue)

# Hiển thị kênh màu Xanh lá (Green) - kênh thứ 2 (chỉ số 1)
green_channel = data[:,:,1]
plt.imshow(green_channel, cmap='gray') # Hiển thị kênh đơn dưới dạng ảnh xám
plt.title("Kênh màu Xanh lá (Green)")
plt.show()

# Hiển thị kênh màu Xanh lam (Blue) - kênh thứ 3 (chỉ số 2)
blue_channel = data[:,:,2]
plt.imshow(blue_channel, cmap='gray') # Hiển thị kênh đơn dưới dạng ảnh xám
plt.title("Kênh màu Xanh lam (Blue)")
plt.show()

# Trong tài liệu gốc, có vẻ họ cộng kênh xanh lá và xanh lam.
# Điều này có thể được dùng để tạo một ảnh xám tổng hợp từ 2 kênh.
bdata = (data[:,:,1].astype(np.float32) + data[:,:,2].astype(np.float32)) / 2.0 # Đảm bảo kiểu float cho phép cộng, sau đó chia trung bình
bdata = bdata.astype(np.uint8) # Chuyển lại về uint8 để hiển thị đúng
plt.imshow(bdata, cmap='gray')
plt.title("Tổng hợp kênh Xanh lá và Xanh lam")
plt.show()
import numpy as np
import imageio.v2 as iio
import matplotlib.pylab as plt
import colorsys

print("--- 2.5 Màu sắc với hệ HSV ---")

# Ví dụ chuyển đổi RGB sang HSV và HSV sang RGB cho một pixel
# Lưu ý: Hàm colorsys mong đợi giá trị RGB/HSV trong khoảng [0, 1]
# Bạn cần chuẩn hóa dữ liệu ảnh (0-255) về 0-1 trước khi dùng colorsys,
# và chuyển đổi lại về 0-255 nếu muốn hiển thị hoặc lưu.

# Ví dụ cho một màu đơn lẻ (Đỏ)
rgb_pixel_red = (255, 0, 0)
hsv_red = colorsys.rgb_to_hsv(rgb_pixel_red[0]/255, rgb_pixel_red[1]/255, rgb_pixel_red[2]/255)
print(f"RGB Đỏ {rgb_pixel_red} -> HSV {hsv_red}")

# Ví dụ cho một màu đơn lẻ (Xanh lam)
rgb_pixel_blue = (0, 0, 255)
hsv_blue = colorsys.rgb_to_hsv(rgb_pixel_blue[0]/255, rgb_pixel_blue[1]/255, rgb_pixel_blue[2]/255)
print(f"RGB Xanh lam {rgb_pixel_blue} -> HSV {hsv_blue}")

# Ví dụ cho một màu đơn lẻ (Xanh lá cây)
rgb_pixel_green = (0, 255, 0)
hsv_green = colorsys.rgb_to_hsv(rgb_pixel_green[0]/255, rgb_pixel_green[1]/255, rgb_pixel_green[2]/255)
print(f"RGB Xanh lá cây {rgb_pixel_green} -> HSV {hsv_green}")

# Ví dụ chuyển đổi HSV sang RGB
hsv_pixel_green_reconstruct = (1/3, 1, 1) # Xanh lá cây trong HSV
rgb_reconstruct = colorsys.hsv_to_rgb(hsv_pixel_green_reconstruct[0], hsv_pixel_green_reconstruct[1], hsv_pixel_green_reconstruct[2])
print(f"HSV {hsv_pixel_green_reconstruct} -> RGB {rgb_reconstruct}")

# Ứng dụng chuyển đổi hệ màu cho toàn bộ ảnh (Bài tập 2.6)
print("--- Ứng dụng chuyển đổi hệ màu cho toàn bộ ảnh (Bài tập 2.6) ---")
data_rgb = iio.imread('bird.png')
height, width, _ = data_rgb.shape

# Chuẩn hóa dữ liệu ảnh về khoảng [0, 1]
data_rgb_norm = data_rgb.astype(np.float32) / 255.0

# Chuyển đổi từng pixel từ RGB sang HSV
# Sử dụng vòng lặp hoặc np.vectorize (np.vectorize có thể chậm với mảng lớn)
# Cách hiệu quả hơn là tự viết hàm chuyển đổi mảng.
# Tuy nhiên, để bám sát ví dụ và tính đơn giản, ta dùng vòng lặp cho mỗi pixel hoặc áp dụng các hàm đã có.

# Cách tốt nhất để chuyển đổi toàn bộ ảnh giữa RGB và HSV bằng NumPy là viết hàm tự mình
# hoặc sử dụng thư viện như scikit-image's rgb2hsv/hsv2rgb nếu có.
# Dưới đây là cách thủ công để bạn hiểu rõ hơn:

h_channel = np.zeros((height, width), dtype=np.float32)
s_channel = np.zeros((height, width), dtype=np.float32)
v_channel = np.zeros((height, width), dtype=np.float32)

for y in range(height):
    for x in range(width):
        r, g, b = data_rgb_norm[y, x]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h_channel[y, x] = h
        s_channel[y, x] = s
        v_channel[y, x] = v

# Bài tập yêu cầu: "thay thế giá trị kênh Hue của các pixel trong một ảnh bằng bằng phương giá trị ban đầu của Hue"
# Giả sử điều này có nghĩa là thay đổi kênh Hue dựa trên giá trị gốc của nó, ví dụ: nhân với một hệ số.
# Ví dụ: Giảm giá trị Hue xuống 1/3 (Knew = 1/3 Kold)
h_modified_channel = h_channel / 3.0
# Đảm bảo H vẫn nằm trong khoảng [0, 1] (hoặc [0, 360) tùy cách định nghĩa)
# Nếu giá trị H giảm, màu sắc sẽ dịch chuyển.

# Tạo lại ảnh HSV mới từ H đã sửa đổi và S, V gốc
# Kết hợp các kênh lại thành một mảng 3D
hsv_modified_image_norm = np.stack((h_modified_channel, s_channel, v_channel), axis=-1)

# Chuyển đổi ảnh HSV đã sửa đổi trở lại RGB
rgb_modified_image_norm = np.zeros_like(data_rgb_norm)
for y in range(height):
    for x in range(width):
        h, s, v = hsv_modified_image_norm[y, x]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        rgb_modified_image_norm[y, x] = [r, g, b]

# Chuyển đổi lại về khoảng [0, 255] và kiểu uint8
rgb_modified_image = (rgb_modified_image_norm * 255).astype(np.uint8)

iio.imwrite('bird_hsv_modified.png', rgb_modified_image)
print("Đã lưu ảnh 'bird_hsv_modified.png' với kênh Hue đã sửa đổi.")
plt.imshow(rgb_modified_image)
plt.title("Ảnh với kênh Hue đã sửa đổi")
plt.show()
import numpy as np
import imageio.v2 as iio
import scipy.ndimage as sn # Để sử dụng các bộ lọc
import matplotlib.pylab as plt
# import colorsys # Không cần cho phần này

print("--- 2.7 Lọc ảnh ---")

# Mở ảnh và chuyển sang ảnh xám
a = iio.imread('bird.png', as_gray=True)
a = a.astype(np.float32) # Chuyển sang float để tính toán lọc dễ dàng hơn

# --- Lọc ảnh với bộ lọc trung bình (Mean filter) ---
print("--- Lọc trung bình (Mean filter) ---")
# Khởi tạo bộ lọc kích thước 5x5
# Bộ lọc được chia cho 25 để chuẩn hóa (tổng các phần tử bằng 1)
k_mean = np.ones((5,5))/25

# Thực hiện phép chập (convolution)
b_mean = sn.convolve(a, k_mean)
b_mean = np.clip(b_mean, 0, 255).astype(np.uint8) # Đảm bảo giá trị nằm trong 0-255 và chuyển về uint8
iio.imwrite('bird_mean_filter.png', b_mean)
print("Đã lưu ảnh 'bird_mean_filter.png'.")
plt.imshow(b_mean, cmap='gray')
plt.title("Ảnh sau lọc trung bình")
plt.show()

# --- Lọc ảnh với bộ lọc trung vị (Median filter) ---
print("--- Lọc trung vị (Median filter) ---")
# size=5 là kích thước cửa sổ 5x5
# mode='reflect': cách xử lý biên ảnh (phản xạ giá trị tại biên)
b_median = sn.median_filter(a, size=5, mode='reflect')
b_median = np.clip(b_median, 0, 255).astype(np.uint8)
iio.imwrite('bird_median_filter.png', b_median)
print("Đã lưu ảnh 'bird_median_filter.png'.")
plt.imshow(b_median, cmap='gray')
plt.title("Ảnh sau lọc trung vị")
plt.show()

# --- Lọc ảnh với bộ lọc cực đại (Max filter) ---
print("--- Lọc cực đại (Max filter) ---")
# Tăng độ sáng cho ảnh. Giá trị lớn nhất trong phụ sẽ thay thế cho giá trị (i,j).
b_max = sn.maximum_filter(a, size=5, mode='reflect')
b_max = np.clip(b_max, 0, 255).astype(np.uint8)
iio.imwrite('bird_max_filter.png', b_max)
print("Đã lưu ảnh 'bird_max_filter.png'.")
plt.imshow(b_max, cmap='gray')
plt.title("Ảnh sau lọc cực đại")
plt.show()

# --- Lọc ảnh với bộ lọc cực tiểu (Min filter) ---
print("--- Lọc cực tiểu (Min filter) ---")
# Kỹ thuật này làm tối độ sáng cho ảnh. Giá trị nhỏ nhất trong phụ sẽ thay thế cho giá trị (i,j).
b_min = sn.minimum_filter(a, size=5, mode='reflect')
b_min = np.clip(b_min, 0, 255).astype(np.uint8)
iio.imwrite('bird_min_filter.png', b_min)
print("Đã lưu ảnh 'bird_min_filter.png'.")
plt.imshow(b_min, cmap='gray')
plt.title("Ảnh sau lọc cực tiểu")
plt.show()
import numpy as np
import imageio.v2 as iio
import scipy.ndimage as sn
from skimage import filters, feature # filters cho Sobel, Prewitt; feature cho Canny
import matplotlib.pylab as plt

print("--- 2.8 Phát hiện biên của ảnh ---")

# Mở ảnh và chuyển sang ảnh xám
a = iio.imread('bird.png', as_gray=True)
a = a.astype(np.float32) # Chuyển sang float để tính toán bộ lọc

# --- Phát hiện biên bằng bộ lọc Sobel ---
print("--- Phát hiện biên bằng bộ lọc Sobel ---")
# filters.sobel trả về ảnh với các giá trị biên (có thể là float)
b_sobel = filters.sobel(a)
# Chuẩn hóa về 0-255 và chuyển sang uint8 để hiển thị và lưu
b_sobel = (b_sobel / b_sobel.max() * 255).astype(np.uint8) if b_sobel.max() > 0 else np.zeros_like(b_sobel, dtype=np.uint8)

iio.imwrite('bird_sobel_filter_edge_detection.png', b_sobel)
print("Đã lưu ảnh 'bird_sobel_filter_edge_detection.png'.")
plt.imshow(b_sobel, cmap='gray')
plt.title("Biên ảnh bằng Sobel")
plt.show()

# --- Phát hiện biên bằng bộ lọc Prewitt ---
print("--- Phát hiện biên bằng bộ lọc Prewitt ---")
b_prewitt = filters.prewitt(a)
b_prewitt = (b_prewitt / b_prewitt.max() * 255).astype(np.uint8) if b_prewitt.max() > 0 else np.zeros_like(b_prewitt, dtype=np.uint8)

iio.imwrite('bird_prewitt_filter_edge_detection.png', b_prewitt)
print("Đã lưu ảnh 'bird_prewitt_filter_edge_detection.png'.")
plt.imshow(b_prewitt, cmap='gray')
plt.title("Biên ảnh bằng Prewitt")
plt.show()

# --- Phát hiện biên bằng bộ lọc Canny ---
print("--- Phát hiện biên bằng bộ lọc Canny ---")
# feature.canny trả về ảnh boolean (True/False cho biên), cần chuyển sang uint8 (0/255)
b_canny = feature.canny(a, sigma=3) # sigma là độ lệch chuẩn của bộ lọc Gaussian dùng để làm mờ trước
b_canny_uint8 = (b_canny * 255).astype(np.uint8) # Chuyển True thành 255, False thành 0

iio.imwrite('bird_canny_filter_edge_detection.png', b_canny_uint8)
print("Đã lưu ảnh 'bird_canny_filter_edge_detection.png'.")
plt.imshow(b_canny_uint8, cmap='gray')
plt.title("Biên ảnh bằng Canny")
plt.show()

# --- Phát hiện biên bằng bộ lọc Laplacian ---
print("--- Phát hiện biên bằng bộ lọc Laplacian (Second derivative) ---")
# sn.laplace trả về giá trị có thể âm, cần chuẩn hóa
b_laplace = sn.laplace(a, mode='reflect')
# Để hiển thị, thường lấy giá trị tuyệt đối hoặc chuẩn hóa lại
b_laplace_display = np.abs(b_laplace)
b_laplace_display = (b_laplace_display / b_laplace_display.max() * 255).astype(np.uint8) if b_laplace_display.max() > 0 else np.zeros_like(b_laplace_display, dtype=np.uint8)

iio.imwrite('bird_laplace_filter_edge_detection.png', b_laplace_display)
print("Đã lưu ảnh 'bird_laplace_filter_edge_detection.png'.")
plt.imshow(b_laplace_display, cmap='gray')
plt.title("Biên ảnh bằng Laplacian")
plt.show()
