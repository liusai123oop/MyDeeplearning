from draw_loss import draw_loss

train_losses = [1.352, 1.190, 1.111, 1.029, 0.960]  # 用于保存每个epoch的训练损失
val_accuracies = [0.456, 0.529, 0.588, 0.621, 0.610]  # 用于保存每个epoch的验证准确率
if __name__ == '__main__':
    draw_loss(5, train_losses, val_accuracies, save_path='logs/loss.png')
