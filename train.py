# Пример: train / val / test, model.train() / model.eval() и метрики в PyTorch
# Запуск: pip install torch scikit-learn && python train.py
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

torch.manual_seed(0)

# 1. Искусственные данные: 1000 клиентов, 5 признаков, метка 0/1 (например, уйдёт или нет)
X = torch.randn(1000, 5)
y = ((X[:, 0] + X[:, 1] - X[:, 2] + 0.5 * torch.randn(1000)) > 0).float().unsqueeze(1)
ds = TensorDataset(X, y)

# 2. Делим на train / val / test (70 / 15 / 15)
train_ds, val_ds, test_ds = random_split(ds, [700, 150, 150])
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=150)
test_dl = DataLoader(test_ds, batch_size=150)

# 3. Модель с Dropout (он ведёт себя по-разному в train и eval)
model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1), nn.Sigmoid())
loss_fn = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

def evaluate(dl):
    model.eval()                      # режим проверки: Dropout выключен
    with torch.no_grad():             # градиенты не нужны
        xb, yb = next(iter(dl))
        p = model(xb)
        loss = loss_fn(p, yb).item()
        pred = (p > 0.5).int().numpy()
    return loss, yb.int().numpy(), pred

best_val, best_state = 1e9, None
for epoch in range(1, 31):
    model.train()                     # режим обучения: Dropout включен
    total = 0
    for xb, yb in train_dl:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
    val_loss, _, _ = evaluate(val_dl)
    print(f"Эпоха {epoch:2d} | train loss {total/700:.3f} | val loss {val_loss:.3f}")
    if val_loss < best_val:           # запоминаем лучшую эпоху по val
        best_val, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}

# 4. Финальная проверка на test (один раз!)
model.load_state_dict(best_state)
_, true, pred = evaluate(test_dl)
print("\nTEST")
print("Accuracy :", round(accuracy_score(true, pred), 3))
print("Precision:", round(precision_score(true, pred), 3))
print("Recall   :", round(recall_score(true, pred), 3))
print("F1       :", round(f1_score(true, pred), 3))
