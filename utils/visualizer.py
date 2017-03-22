import matplotlib.pyplot as plt
import numpy as np

class Visualizer():
    def __init__(self,
                 num_epochs,
                 metrics_template,
                 watches):
        self.watches = watches
        self.num_epochs = num_epochs
        
        self.train_metrics = {}
        self.val_metrics = {}
        for metric_name, metric in metrics_template.items():
            if metric_name == 'complexity_parts':
                self.train_metrics[metric_name] = [None] * len(metric)
                self.val_metrics[metric_name] = [None] * len(metric)
                
                for i in range(len(metric)):
                    self.train_metrics[metric_name][i] = []
                    self.val_metrics[metric_name][i] = []
            else:
                self.train_metrics[metric_name] = []
                self.val_metrics[metric_name] = []
                
        self.fig1, ax1 = plt.subplots(1, 1)
        self.fig2, ax2 = plt.subplots(1, 1)
        ax1.set_xlabel('Num Epoch')
        ax1.set_ylabel('Metric Value')
        ax1.set_xlabel('Num Epoch')
        ax1.set_ylabel('Metric Value')
        ax2.set_xlabel('Num Epoch')
        ax2.set_ylabel('Metric Value')
        self.lines1 = {}
        self.lines2 = {}
        
        if 'total_complexity' in watches or 'complexity_parts' in watches:
            self.fig3, ax3 = plt.subplots(1, 1)
            self.fig4, ax4 = plt.subplots(1, 1)
            ax3.set_xlabel('Num Epoch')
            ax3.set_ylabel('Train Complexity')
            ax4.set_xlabel('Num Epoch')
            ax4.set_ylabel('Val Complexity')
            self.lines3 = {}
            self.lines4 = {}
    
        self.template_space = np.zeros(num_epochs)
        
        for watch in watches:            
            if watch == 'total_complexity':
                self.lines3[watch], = ax3.plot(np.arange(num_epochs), self.template_space, label='Train ' + watch)
                self.lines4[watch], = ax4.plot(np.arange(num_epochs), self.template_space, label='Val ' + watch)
            elif watch == 'complexity_parts':
                self.lines3[watch] = [None] * len(metrics_template[watch])
                self.lines4[watch] = [None] * len(metrics_template[watch])
                
                for j in range(len(metrics_template[watch])):
                    self.lines3[watch][j], = ax3.plot(np.arange(num_epochs),
                                                      self.template_space,
                                                      label='Train ' + watch + str(j + 1))
                    self.lines4[watch][j], = ax4.plot(np.arange(num_epochs),
                                                      self.template_space,
                                                      label='Val ' + watch + str(j + 1))
            else:
                self.lines1[watch], = ax1.plot(np.arange(num_epochs), self.template_space, label='Train ' + watch)
                self.lines2[watch], = ax2.plot(np.arange(num_epochs), self.template_space, label='Val ' + watch)
    
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax1.set_ylim(0, 1.01)
        ax2.set_ylim(0, 1.01)

        self.fig1.show()
        self.fig2.show()
        
        if 'total_complexity' in watches or 'complexity_parts' in watches:
            ax3.legend(loc='best')
            ax4.legend(loc='best')
            ax3.set_ylim(0, 1.01)
            ax4.set_ylim(0, 1.01)

            self.fig3.show()
            self.fig4.show()
        
    def watch(self, train_measurements, val_measurements):
        for metric_name, metric in train_measurements.items():
            if metric_name == 'complexity_parts':
                for j in range(len(self.train_metrics[metric_name])):
                    self.train_metrics[metric_name][j].append(metric[j])
                    self.val_metrics[metric_name][j].append(val_measurements[metric_name][j])
            else:
                self.train_metrics[metric_name].append(metric)
                self.val_metrics[metric_name].append(val_measurements[metric_name])
            
        for watch in self.watches:                       
            if watch == 'total_complexity':
                template_y3 = self.template_space.copy()
                template_y4 = self.template_space.copy()
                
                template_y3[:len(self.train_metrics[watch])] = self.train_metrics[watch]
                template_y4[:len(self.val_metrics[watch])] = self.val_metrics[watch]

                self.lines3[watch].set_ydata(template_y3)
                self.lines4[watch].set_ydata(template_y4)
            elif watch == 'complexity_parts':
                for j in range(len(self.train_metrics[watch])):
                    template_y3 = self.template_space.copy()
                    template_y4 = self.template_space.copy()

                    template_y3[:len(self.train_metrics[watch][j])] = self.train_metrics[watch][j]
                    template_y4[:len(self.val_metrics[watch][j])] = self.val_metrics[watch][j]
                    
                    self.lines3[watch][j].set_ydata(template_y3)
                    self.lines4[watch][j].set_ydata(template_y4)
            else:
                template_y1 = self.template_space.copy()
                template_y2 = self.template_space.copy()
                
                template_y1[:len(self.train_metrics[watch])] = self.train_metrics[watch]
                template_y2[:len(self.val_metrics[watch])] = self.val_metrics[watch]

                self.lines1[watch].set_ydata(template_y1)
                self.lines2[watch].set_ydata(template_y2)
        
        self.fig1.canvas.draw()       
        self.fig2.canvas.draw()
        
        if 'total_complexity' in self.watches or 'complexity_parts' in self.watches:
            self.fig3.canvas.draw()       
            self.fig4.canvas.draw()
        
    def finish(self):
        plt.close(self.fig1)
        plt.close(self.fig2)
        
        if 'total_complexity' in self.watches or 'complexity_parts' in self.watches:
            plt.close(self.fig3)
            plt.close(self.fig4)
        