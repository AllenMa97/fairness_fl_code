import torch
import torch.nn as nn

class Standard_LoRA_Layer(nn.Module):
    def __init__(self, input_size, output_size, rank, alpha):
        super(Standard_LoRA_Layer, self).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.rand(input_size, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, output_size))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class Heavy_LoRA_Layer(nn.Module):
    def count_divisions(self, A, B):
        count = 0
        while A > B:
            A = A // 2  # 使用整数除法
            count += 1
        # 注意：循环结束时，A 可能已经小于或等于 B，但我们只关心达到这个条件之前的次数
        return count

    def __init__(self, input_size, output_size, rank, alpha):
        super(Heavy_LoRA_Layer, self).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        A_times = self.count_divisions(input_size, rank)
        self.A_list = []
        I = input_size
        for i in range(0, A_times, 1):
            if i != A_times-1:
                O = I // 2
                std_dev = 1 / torch.sqrt(torch.tensor(O).float())
                A = torch.nn.Parameter(torch.rand(I, O) * std_dev)
                self.A_list.append(A)
                I = O
            else:
                std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
                A = torch.nn.Parameter(torch.rand(I, rank) * std_dev)
                self.A_list.append(A)

        self.B_list = []
        B_times = self.count_divisions(output_size, rank) + 1
        for i in range(B_times, 0, -1):
            if i == B_times:
                B = torch.nn.Parameter(torch.zeros(rank, I))
                self.B_list.append(B)
            elif i != 0:
                O = I * 2
                B = torch.nn.Parameter(torch.zeros(I, O))
                self.B_list.append(B)
                I = O
            else:
                B = torch.nn.Parameter(torch.zeros(I, output_size))
                self.B_list.append(B)

        self.alpha = alpha

    def forward(self, x):
        for item in self.A_list:
            x = x @ item
        for item in self.B_list:
            x = x @ item
        x = self.alpha * x
        return x

class Linear_With_LoRA(nn.Module):
    def __init__(self, linear_layer, rank, alpha, LoRa_type='stander'):
        super().__init__()
        self.linear = linear_layer
        if 'heavy' in LoRa_type:
            self.LoRA_func = Heavy_LoRA_Layer
        else:
            self.LoRA_func = Standard_LoRA_Layer

        self.LoRA = self.LoRA_func(input_size=self.linear.in_features,
                                     output_size=self.linear.out_features,
                                     rank=rank, alpha=alpha)

    def forward(self, x):
        return self.linear(x) + self.LoRA

if __name__ == '__main__':
    a = torch.ones([3, 128])

    mo = Standard_LoRA_Layer(128, 64, 32, 2)
    y = mo(a)

    mo2 = Heavy_LoRA_Layer(128, 64, 32, 2)
    y2 = mo2(a)

    print(2)