import tensorflow as tf

X = tf.constant([[1,2,3],
                 [4,5,6],
                 [7,8,9]
                ])

print(X)

Y = tf.constant([[10,11],
                 [12,13],
                 [14,15]])

print(Y)

print("Multiplication : ",X*X)
try :
    print("Multiplication : ",X*Y)
except :
    print("Normal Mukltiplication works with same shape tensors only.")

print("Matrix multiplication(Dot product) : ", tf.matmul(X,Y))

print(tf.tensordot(X,Y,axes=0))

