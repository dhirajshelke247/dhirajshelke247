n=int (input("enter prime number(n)"))
g=int (input("enter base prime number(g)"))
alicesecret=int(input("enter alice secret key(x)"))
bobsecret=int (input("enter bob secret key(y)"))
print("publicly shared prime(n):",n)
print("publicly shared prime(g):",g)
print("alice secret key:",alicesecret)
print("bob secret key:",bobsecret)
A=(g**alicesecret)%n
print("\n Alice sends (A) to Bob:",A)
B=(g**bobsecret)%n
print("\n Bob sends (B) to Alice:",B)
alicesecretkey=(B**alicesecret)%n
print("\n Alice shared secret:",alicesecretkey)
bobsecretkey=(A**bobsecret)%n
print("\n Bob shared secret:",bobsecretkey)
