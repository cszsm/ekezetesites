import tensorflow as tf

from sklearn.feature_extraction import DictVectorizer
import preparer
import helper

################

vectorizer = DictVectorizer()
vectorizer.fit(helper.generate_windows(4))

text = "A szélsőjobbos alt-right mozgalom Trump közelsége nem volt kérdéses, köszönhetően olyan lépéseknek, mint hogy Trump az alt-right egyik kedvelt lapjának, a Breitbartnak a vezérigazgatóját választotta kampányfőnökének és főstratégájának. Az ügy most vált különösen kényessé, amikor előkerült egy felvétel,amin az alt-right vezetője, a náci ideológiák terjesztése miatt Magyarországról is kitiltott Richard B. Spencer 'Hail Trump'-felkiáltással zárja egy alt-rightos konferenciáját. A résztvevők pedig náci karlendítéssel válaszoltak. Trump a felvételre úgy válaszolt: 'Megtagadom, és elítélem őket.' A Breitbart főnökének kritikájára pedig érzelmes húrokat pengetett meg. Bannont szerinte nagyon megviselte, hogy a nácikkal emlegették egy lapon a mainstream amerikai médiában. Sajátos logikával bizonyította, hogy Bannon nem szélsőjobbos: 'Ha azt gondolom, hogy ő náci, alt-rightos, vagy hasonló, akkor biztos hogy nem alkalmazom őt.' - érvelt. A Hollywood Reporter egyébként összeszedte, hogy mely más nyelvű lapok a Breitbart megfelelői. Szerintük a Kurucinfo a magyar Breitbart."
x_e, y_e = preparer.prepare_text(text, 4, "e")
vectorized_x = vectorizer.transform(x_e).toarray()

test = "Tévedsz. Eddig epedtek érte, hogy legyen, s nem volt, most majd a lelkük üdvösségét kínálnák, ha elmaradhatna, de nem tudjuk megakadályozni."
test_x, test_y = preparer.prepare_text(test, 4, "e")
vectorized_test = vectorizer.transform(test_x).toarray()

f = open("text2.txt")
text4 = f.read()
text4_x, text4_y = preparer.prepare_text(text4, 4, "e")
v_text4_x = vectorizer.transform(text4_x).toarray()


################

input_size = len(vectorized_x[0])
output_size = 2

n_input = tf.placeholder(tf.float32, [None, input_size])
n_output = tf.placeholder(tf.float32, [None, output_size])

hidden_neurons = 10

b_hidden = tf.Variable(tf.random_normal([hidden_neurons]))
W_hidden = tf.Variable(tf.random_normal([input_size, hidden_neurons]))
hidden = tf.sigmoid(tf.matmul(n_input, W_hidden) + b_hidden)

W_output = tf.Variable(tf.random_normal([hidden_neurons, output_size]))
output = tf.sigmoid(tf.matmul(hidden, W_output))

# y = tf.nn.softmax(tf.matmul(n_input, W_hidden) + b_hidden)

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(n_output * tf.log(y), reduction_indices=[1]))
cost = tf.reduce_mean(tf.square(n_output - output))

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# for i in range(1000):
#     sess.run(train_step, feed_dict={n_input: v_text4_x, n_output: text4_y})

for i in range(1000):
    cvalues = sess.run([train, cost, W_hidden, b_hidden, W_output], feed_dict={n_input: v_text4_x, n_output: text4_y})

    if i % 100 == 0:
        print("")
        print("step: {:>3}".format(i))
        print("loss: {}".format(cvalues[1]))

print("")
print(sess.run(output, feed_dict={n_input: vectorized_test}))

# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(n_output, 1))
#
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print(sess.run(accuracy, feed_dict={x: v_text4_x, y_: text4_y}))
# print(sess.run(accuracy, feed_dict={n_input: vectorized_test, n_output: test_y}))
# word = "Eddig epedtek"
# vx, vy = preparer.prepare_text(word, 4, "e")
# print(sess.run(accuracy, feed_dict={x: vectorizer.transform(vx).toarray(), y_: [[1, 0], [0, 1], [1, 0], [1, 0]]}))