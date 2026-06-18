import torch
import matplotlib.pyplot as plt

# 1. Tải checkpoint cũ (Nhớ thay đường dẫn file của bạn)
checkpoint_path = r"D:\data_leak\best_iou_model.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 2. Tìm state_dict (tùy cách bạn lưu, có thể nó nằm trong key 'state_dict' hoặc 'model_state_dict')
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint # Trường hợp bạn lưu thẳng model.state_dict()

# 3. Mò tìm cái key chứa prototype (vì nếu dùng DataParallel nó có thể có tiền tố 'module.')
prototype_key = [k for k in state_dict.keys() if 'base_prototypes' in k][0]
optimal_prototype = state_dict[prototype_key]

print(f"Đã tìm thấy Prototype tại key: {prototype_key}")
print(f"Kích thước ma trận (Shape): {optimal_prototype.shape}")

# 4. Trực quan hóa nó bằng Heatmap (Để xem nó phân bố ra sao)
plt.figure(figsize=(10, 2))
plt.imshow(optimal_prototype.numpy(), aspect='auto', cmap='viridis')
plt.colorbar(label='Giá trị Vector')
plt.title('Bản đồ nhiệt của Prototype Tối Ưu (62.40% mIoU)')
plt.xlabel('256 Chiều Không Gian (Embedding Dim)')
plt.ylabel('Class')
plt.show()

# 5. Lưu nó ra một file riêng để tiện dùng lại
# torch.save(optimal_prototype, 'optimal_coral_prototype.pth')
# print("Đã xuất khẩu thành công Prototype tối ưu!")