import numpy as np
import matplotlib.pyplot as plt

N = 10
men_means = (20, 35, 30, 35, 27, 20, 35, 30, 35, 27)
men_std = (2, 3, 4, 1, 2, 3, 5, 2, 3, 3)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
#rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)
rects1 = ax.bar(ind, men_means, width)

#women_means = (25, 32, 34, 20, 25, 20, 35, 30, 35, 27)
#women_std = (3, 5, 2, 3, 3, 3, 5, 2, 3, 3)
#rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Labels')
ax.set_title('Test data labels')
ax.set_xticks(ind + width / 2)
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

#ax.legend((rects1[0], rects2[0]), ('Training', 'Test'))

"""
def autolabel(rects):
"""
#Attach a text label above each bar displaying its height
"""
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
"""
#autolabel(rects1)
#autolabel(rects2)


plt.show()