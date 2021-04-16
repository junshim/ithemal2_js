import os

HOME1 =os.path.dirname(os.path.abspath(__file__))
HOME2 =os.path.join(HOME1,'..')
#HOME3 =os.path.abspath(os.path.dirname(HOME1))
HOME3 =os.path.dirname(HOME1)

print(HOME1)
print(HOME2)
print(HOME3)

