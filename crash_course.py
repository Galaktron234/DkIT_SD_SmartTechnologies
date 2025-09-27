import numpy as np

def main():
    stacking()


def arithmetic_operators():
    print(1 + 1)
    print(1 - 1)
    print(5 * 10)
    print(10 / 3)
    print(6 + 9 * 3 + 3)
    print(6 % 2)
    print(13 // 2)
    print(6 ^ 2)
    
    
def variables():
    age = 25_000_000_000_000_000
    name = "John"
    gender = "male"
    
    age += 5
    print(age)
    
    
def numeric_data_types():
    x = 7
    y = 3
    
    z = x / y
    print(type(x))
    
    print(2 + 2.5)
    print(int(7 / 3.5))
    
    w = int(3.7)
    
    w_float = float(3)
    
    
def strings():
    name = "harjappan"
    print(type(name))
    
    dialogue = 'Harjappan said "10 / 3 is 3"'
    
    dialogue = 'Harjappan said "10 / 3 is 3" to which John replied "you\'re having a bad day'
    
    print(dialogue * 10)
    print(len(dialogue))
    
    
def booleans():
    harjappan_is_having_a_good_day = True
    comparison_operation = 1 < 2 and 2 > 3
    print(comparison_operation)
    
    
def methods():
    movie_title = "Star Wars"
    print(movie_title.upper())
    print(movie_title.count('a'))
    
    
def lists():
    names = ["Alexander", "Haroldas", "Conor"]
    print(names[0])
    
    try:
        print(names[100])
    except IndexError:
        print("Nothing at that index")
        
    random_variables = [True, False, "Hello", 1, 1.2]
    length = len(random_variables)
    print(random_variables[-1])
    
    
def slicing_lists():
    ordered_numbers = list(range(0, 10, 2))
    print(ordered_numbers)
    print(ordered_numbers[2 : 10])
    print(ordered_numbers[ : 10])
    print(ordered_numbers[2 : ])
    
    show_title = "the office"
    show_title[2]
    show_title[4: ]


def membership_operators():
    months = ["January", "February", "March"]
    print("January" in months)
    print("June" in months)  
    
    course = "Software Development"
    print("Software" in course)  
    
    
def mutability():
    # Mutable = changeable, can an object be changed once it has been created
    # Immutable = cannot be changed without creating a completely new object
    grocery_list = ['eggs', 'bread', 'milk']
    print(grocery_list)
    grocery_list[2] = "chocolate"
    print(grocery_list)
    
    misspelled_vegetable = "bwoccoli"
    misspelled_vegetable = "broccoli"
    print(misspelled_vegetable)
    
    name = "Elga"
    other_name = name
    name = "Mila"
    print(name)
    print(other_name)
    
    books = ['Atomic Habits', 'Percy Jackson', 'Rich Dad, Poor Dad']
    more_books = books
    books[0] = "A Handmaid's Tale"
    print(books)
    print(more_books) 
    
    
def functions_for_lists():
    numbers = [1, 2, 3, 4, 5]
    print(len(numbers))
    print(max(numbers))
    names = ["Martin", "Diane", "Yee Chen"]
    print(sorted(names))
    
    print('-'.join(["Jan", "Feb", "Mar"]))
    
    names.append("Luke")
    print(f"The list is names is {names}")
    
    
def tuples():
    traits = ('tall', 'kind', 'intelligent')
    print(traits[0])
    height, personality, iq = traits
    
    
def test_sets():
    # Not ordered, mutable, no duplicates
    duplicate_numbers = [1, 1, 2, 2, 3, 3]
    unique_numbers = set(duplicate_numbers)   
    print(unique_numbers) 
    unique_numbers.add(4)
    unique_numbers.add(3)
    print(2 in unique_numbers)
    

def test_dictionary():
    # Mutable, Not ordered, key value pairs
    # keys have to be unique and immutable
    inventory = {'bananas' : 2.00, 'apples' : 0.60, 'grapes' : 2.50}
    inventory["bananas"] = 1.79
    price = inventory.get("bananas")
    if("strawberry" in inventory):
        strawberry_price = inventory["strawberry"]
        print(strawberry_price)
        
        
def compound_ds():
    grocery_items = {'bananas' : {'price' : 1.79, 'country of origin' : 'Thailand'}, 'apples' : {'price' : 0.60, 'country of origin' : 'Ireland'}}
    print(grocery_items["apples"])
    print(grocery_items["apples"]["country of origin"])
    
    
def pythonic_while():
    while True:
        number = int(input("Enter even number: "))
        if number % 2 == 0:
            print("Even")
            break
        
        
def doc_strings():
    # Comment to explain a function
    """ Testing out doc string"""
    """
     INPUT: None
     OUTPUT: Name
    """
 

def even_or_odd(number):
    return number % 2 == 0   

   
def higher_order_function():
    numbers = [1, 2, 3, 4, 5]
    print(list(filter(even_or_odd, numbers)))
    
    
def lambda_function():
    numbers = [1, 2, 3, 4, 5]
    print(list(filter(lambda number: number % 2 == 0, numbers)))
    
    
# Numpy crash course

def numpy_v_list():
    list_two = list(range(1, 4))
    list_three = list(range(1, 4))
    list_sum = []
    
    for index in range(3):
        list_two[index] = list_two[index] ** 2
        list_three[index] = list_three[index] ** 3
        list_sum.append(list_two[index] + list_three[index])

    print(list_sum)
    
    array_two = np.arange(1, 4) ** 2
    array_three = np.arange(1, 4) ** 3
    print(array_two + array_three)
    
    np.power(np.array([1, 2, 3, 4]), 4)
    np.negative(np.array([1, 2, 3, 4]))
    sample_array = np.array([1, 2, 3])
    print(np.exp(sample_array))
    np.log(np.exp(sample_array))
    print(np.sin(sample_array))
    
    
def multi_dimensional_array():
    x = np.arange(3)
    y = np.arange(3)
    z = np.arange(3)
    multi_array = np.array([x, y, z])
    print(multi_array)
    print(multi_array.shape)
    
    w = np.linspace(1, 10, 50)
    print(w)
    
    b = np.arange(1, 30, 3)
    print(b)
    bl = np.linspace(1, 30, 3)
    print(bl)
    wl = np.linspace(1, 30, 100, False)
    print(wl)
    
    
def access_multi_array():
    x = np.arange(3)
    y = np.arange(3, 6)
    z = np.arange(6, 9)
    multi_array = np.array([x, y, z], dtype=np.uint8)
    print(multi_array)
    print(multi_array.dtype)
    
    
def array_slicing():
    x = np.arange(1, 10)
    print(x[2: 7], 2)
    print(x[:7])
    print(x[2:])
    
    
def reshaping():
    x = np.arange(9)
    print(x)
    x = x.reshape(3, 3)
    print(x)
    x = np.arange(18).reshape(3, 2, 3)
    print(x[:, 0, 0])
    
    
def conditional_selection():
    x = np.arange(9)
    comparison_operation = x > 5
    print(comparison_operation)
    print(x[comparison_operation])
    print(x[x > 5])
    print(x.max())
    print(x.min())
    
    
def ravel_and_flatten():
    x = np.arange(9).reshape(3, 3)
    print(x)
    ravelled_array = x.ravel()
    print(ravelled_array)
    
    y = np.arange(9).reshape(3, 3)
    print(y)
    flattened_array = y.flatten()
    print(flattened_array)
    
    ravelled_array[2] = 1_000_000
    print(ravelled_array)
    print(x)
    
    flattened_array[2] = 1_000_000
    print(flattened_array)
    print(y)
    
    
def transpose():
    y = np.arange(9)
    y.shape = [3, 3]
    print(y)
    print(y.transpose())
    print(y.T)
    

def odds_and_ends():
    y = np.arange(9)
    y.shape = [3, 3]
    print(np.resize(y, (6, 5)))
    print(np.zeros((6,), dtype=int))
    print(np.eye(3))
    print(np.random.rand(4, 4))
    
    
def matrix_mult():
    mat_a = np.matrix([0, 3, 5, 5, 5, 2]).reshape(2, 3)
    mat_b = np.matrix([3, 4, 3, -2, 7, 11]).reshape(3, 2)
    print(mat_a * mat_b)
    #product = np.matmul(mat_a, mat_b)
    print(mat_a @ mat_b)
    
    
def stacking():
    x = np.arange(4).reshape(2, 2)
    print(x)
    y = np.arange(4, 12).reshape(4, 2)
    print(y)
    z = np.vstack((x, y))
    print(z)
    print(z.shape)
    

        
        
        
    

if __name__ == "__main__":
    main()
    
    