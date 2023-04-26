import pandas as panda

def merge_lists (list1, list2):

    list3 = list1 + list2
    print ("\nList 1 being merged with List 2 looks like: {0}".format (list3))

    return list3

def sort_list (list_x):

    sorted_list = sorted (list_x)
    print ("\nList 3 sorted looks like: {0}".format (sorted_list))
    return sorted_list


lista = [5, 4, 9, 100, 0, -3]
listb = [-100, 2, 1, -65, 25]

listc = merge_lists (lista, listb)

listc_sorted = sort_list (listc)
