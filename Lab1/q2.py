n=int(input("enter number of elements in list:"))
list=[]

def check_case(n): # function to check range determination
    if n<3:
        return False
    else:
        return True
def take_input(n): # function to take inputs
        for i in range(n):
            a=int(input(f"enter {i+1} element"))
            list.append(a)
    
def sort_list(list): # function to sort list
    list.sort()

def find_range(list): #function to find range 
    min=list[0]
    max=list[-1]
    range=max-min
    return range 

if check_case(n)==False:# printing message if range not possible 
    print("range determination not possible")
while check_case(n)==True:# execute function if range possible 
     take_input(n)
     sort_list(list)
     print("range of list is :",find_range(list))
     break

    
    





    
