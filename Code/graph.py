from graphviz import Graph
import graphviz

#list = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
list = [30]

g = Graph('Vasconcellos Graph', strict = True)

for i in list:
    g.attr('node', shape = 'doublecircle')
    g.node('%02d' % i)

# List append apenas nos filhos dos filhos nas arvores

for i in list:
    if i == 1:
        pass
    if i == 2:
        g.edge('02', '03')
        g.edge('02', '16')
        g.edge('02', '17')
        g.edge('02', '21')
        if (18 not in list): list.append(18)
        if (20 not in list): list.append(20)
    if i == 3:
        g.edge('03', '02')
        if (16 not in list): list.append(16)
        if (21 not in list): list.append(21)
        if (17 not in list): list.append(17)
    if i == 4:
        g.edge('04', '15')
        g.edge('04', '18')
        g.edge('04', '19')
        if (17 not in list): list.append(17)
        if (5 not in list): list.append(5)
        if (15 not in list): list.append(15)
        if (14 not in list): list.append(14)
        if (18 not in list): list.append(18)
    if i == 5:
        g.edge('05', '18')
        if (17 not in list): list.append(17)
        if (4 not in list): list.append(4)
        if (15 not in list): list.append(15)
    if i == 6:
        pass
    if i == 7:
        pass
    if i == 8:
        g.edge('08', '09')
        if (14 not in list): list.append(14)
    if i == 9:
        g.edge('09', '08')
        g.edge('09', '14')
        if (12 not in list): list.append(12)
        if (19 not in list): list.append(19)
        if (29 not in list): list.append(29)
    if i == 10:
        g.edge('10', '11')
        g.edge('10', '30')
        if (25 not in list): list.append(25)
        if (24 not in list): list.append(24)
    if i == 11:
        g.edge('11', '10')
        if (30 not in list): list.append(30)
    if i == 12:
        g.edge('12', '14')
        if (9 not in list): list.append(9)
        if (19 not in list): list.append(19)
        if (29 not in list): list.append(29)
    if i == 13:
        pass
    if i == 14:
        g.edge('14', '09')
        g.edge('14', '12')
        g.edge('14', '19')
        g.edge('14', '29')
        if (4 not in list): list.append(4)
        if (8 not in list): list.append(8)
    if i == 15:
        g.edge('15', '04')
        g.edge('15', '18')
        if (5 not in list): list.append(5)
        if (19 not in list): list.append(19)
        if (18 not in list): list.append(18)
        if (17 not in list): list.append(17)
        if (4 not in list): list.append(4)
    if i == 16:
        g.edge('16', '02')
        if (3 not in list): list.append(3)
        if (17 not in list): list.append(17)
        if (21 not in list): list.append(21)
    if i == 17:
        g.edge('17', '02')
        g.edge('17', '18')
        if (3 not in list): list.append(3)
        if (16 not in list): list.append(16)
        if (21 not in list): list.append(21)
        if (5 not in list): list.append(5)
        if (15 not in list): list.append(15)
        if (4 not in list): list.append(4)
    if i == 18:
        g.edge('18', '04')
        g.edge('18', '05')
        g.edge('18', '15')
        g.edge('18', '17')
        if (2 not in list): list.append(2)
        if (19 not in list): list.append(19)
        if (4 not in list): list.append(4)
    if i == 19:
        g.edge('19', '04')
        g.edge('19', '14')
        if (15 not in list): list.append(15)
        if (18 not in list): list.append(18)
        if (29 not in list): list.append(29)
        if (9 not in list): list.append(9)
        if (12 not in list): list.append(12)
    if i == 20:
        g.edge('20', '21')
        if (2 not in list): list.append(2)
    if i == 21:
        g.edge('21', '02')
        g.edge('21', '20')
        if (16 not in list): list.append(16)
        if (3 not in list): list.append(3)
        if (17 not in list): list.append(17)
    if i == 22:
        g.edge('22', '28')
    if i == 23:
        pass
    if i == 24:
        g.edge('24', '30')
        if (25 not in list): list.append(25)
        if (10 not in list): list.append(10)
    if i == 25:
        g.edge('25', '30')
        if (24 not in list): list.append(24)
        if (10 not in list): list.append(10)
    if i == 26:
        pass
    if i == 27:
        pass
    if i == 28:
        g.edge('28', '22')
    if i == 29:
        g.edge('29', '14')
        if (12 not in list): list.append(12)
        if (19 not in list): list.append(19)
        if (9 not in list): list.append(9)
    if i == 30:
        g.edge('30', '10')
        g.edge('30', '24')
        g.edge('30', '25')
        if (11 not in list): list.append(11)

for i in range(1,31):
    g.attr('node', shape ='circle')
    g.node('%02d' % i)

g.edge('02', '03')
g.edge('08', '09')
g.edge('10', '30')
g.edge('11', '10')
g.edge('14', '09')
g.edge('14', '12')
g.edge('15', '04')
g.edge('15', '18')
g.edge('16', '02')
g.edge('17', '02')
g.edge('17', '18')
g.edge('18', '04')
g.edge('18', '05')
g.edge('19', '04')
g.edge('19', '14')
g.edge('21', '02')
g.edge('21', '20')
g.edge('24', '30')
g.edge('25', '30')
g.edge('28', '22')
g.edge('29', '14')

min_df = 0.4
number_topics = 2
number_words = 10
r = graphviz.Source(g, filename = "graph-with-%0.1f-%d-%d" % (min_df, number_topics, number_words), format = "ps")
r.render()
#r.view()



