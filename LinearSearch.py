def LinearSearch(list, x):
    position = 0
    found = false
    while(position < len(list) and found != false):
        if(list[position] == x):
            found = true
        position = position+1

array=[2,3,6,8,4,10]
print(LinearSearch(array,4))
