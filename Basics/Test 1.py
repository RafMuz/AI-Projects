

def merge_lists (list_a, list_b):

    merged_list = list_a + list_b
    return merged_list

def multiply_lists (list_a, list_b):

    x = 0
    list_d = []

    if len(list_a) == len(list_b):

        while x < len(list_a):

            list_d.append ([list_a [x] * list_b [x]])
            x = x + 1

    return list_d


def even_numbers (list_a):

    x = 0
    list_e = []

    while x < len (list_a):

        if list_a [x] % 2 == 0:

            list_e.append (list_a [x])

        x = x + 1

    return list_e

test_list = [1, 2, 3, 9, 6]
test_list_2 = [1, 2, 3, 9, 6]

list_c = merge_lists (test_list, test_list_2)
list_d = multiply_lists (test_list, test_list_2)

final_list = merge_lists (list_c, list_d)
final_list.sort()

#even_list = even_numbers (sorted_final_list)
#print (list_d)
#print (sorted_final_list)
#print (even_list)