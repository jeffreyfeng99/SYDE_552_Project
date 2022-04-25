from cProfile import label
import os
import matplotlib.pyplot as plt

if __name__=='__main__':
    input_files = ['./pretrained/jeff_work/vgg11/vgg2.txt',
                './pretrained/jeff_work/alif/alif.txt',
                './pretrained/jeff_work/burst/burst.txt'
                ]

    losses = []
    accs = []

    for file in input_files:
        loss = []
        acc = []

        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Training' in line:
                    temp_loss = line.split('Training loss: ')[1]
                    loss.append(float(temp_loss))
                elif 'test_accuracy' in line:
                    temp_acc = line.split('test_accuracy : ')[1]
                    acc.append(float(temp_acc))
        
        losses.append(loss)
        accs.append(acc)

    epochs = list(range(len(losses[0])))
    legend = ['Poisson Generator', 'Adaptive Leaky-Integrate-and-Fire', 'Bursting Generator']

    plt.figure()
    for i in range(len(losses)):
        plt.plot(epochs,losses[i])
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.legend(legend)
        plt.title('Training Loss over Epochs')

    plt.figure()
    for i in range(len(accs)):
        plt.plot(epochs,accs[i])
        plt.xlabel('Epochs')
        plt.ylabel('Testing Accuracy')
        plt.legend(legend)
        plt.title('Validation Accuracy over Epochs')

    plt.show()
