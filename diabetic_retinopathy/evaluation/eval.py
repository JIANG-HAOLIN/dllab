import matplotlib.pyplot as plt

class Plotting():
    def __init__(self):
        self.store_val = []
        self.store_test = []
        self.store_acu_val = []
        self.store_acu_test = []
        self.store_tp = []
        self.store_tn = []
        self.store_fp = []
        self.store_fn = []
        self.store_tp_test = []
        self.store_tn_test = []
        self.store_fp_test = []
        self.store_fn_test = []
        self.cur = []

    def cur_update(self, cur_iter):
        self.cur.append(cur_iter)

    def validation_update(self, loss_val, val_acu, tp, tn, fp, fn):
        self.store_val.append(loss_val.item())
        self.store_acu_val.append(val_acu)

        self.store_tp.append(tp)
        self.store_tn.append(tn)
        self.store_fp.append(fp)
        self.store_fn.append(fn)

    def testing_update(self, loss_val, val_acu, tp, tn, fp, fn):
        self.store_test.append(loss_val.item())
        self.store_acu_test.append(val_acu)

        self.store_tp_test.append(tp)
        self.store_tn_test.append(tn)
        self.store_fp_test.append(fp)
        self.store_fn_test.append(fn)

    def plot_results(self):
        fig = plt.figure()
        plt.plot(self.cur, self.store_val, label="val_loss")  # plot example
        plt.plot(self.cur, self.store_test, label="test_loss")
        plt.legend(loc='upper left')
        fig.savefig('loss.png')

        fig2 = plt.figure()
        plt.plot(self.cur, self.store_acu_val, label="val_accuracy")  # plot example
        plt.plot(self.cur, self.store_acu_test, label="test_accuracy")
        plt.legend()
        fig2.savefig('accuracy.png')

        fig3 = plt.figure()
        plt.plot(self.cur, self.store_tp, label="TP")  # plot example
        plt.plot(self.cur, self.store_tn, label="TN")  # plot example
        plt.plot(self.cur, self.store_fp, label="FP")  # plot example
        plt.plot(self.cur, self.store_fn, label="FN")  # plot example
        plt.legend(loc='upper left')
        fig3.savefig('val_Confusion_Matrix.png')

        fig4 = plt.figure()
        plt.plot(self.cur, self.store_tp_test, label="TP")  # plot example
        plt.plot(self.cur, self.store_tn_test, label="TN")  # plot example
        plt.plot(self.cur, self.store_fp_test, label="FP")  # plot example
        plt.plot(self.cur, self.store_fn_test, label="FN")  # plot example
        plt.legend(loc='upper left')
        fig4.savefig('test_Confusion_Matrix.png')
