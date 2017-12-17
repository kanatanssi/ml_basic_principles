import numpy as np
import matplotlib.pyplot as plt

N = 10
#men_means = (20, 35, 30, 35, 27, 20, 35, 30, 35, 27)

ind = np.arange(N)  # the x locations for the groups
width = 0.40       # the width of the bars

def plot_bar(data, testortrain):
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, data)#, width)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Label count')
    ax.set_title(testortrain+' data labels')
    ax.set_xticks(ind)
    #ax.set_xticklabels(('Pop_Rock',
    #                    'Electronic',
    #                    'Rap',
    #                    'Jazz',
    #                    'Latin',
    #                    'RnB',
    #                    'International',
    #                    'Country',
    #                    'Reggae',
    #                    'Blues'))
    ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))

    #Attach a text label above each bar displaying its height
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            #ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
            #        '%d' % int(height),
            #        ha='center', va='bottom')
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')


    autolabel(rects1)

    plt.show()