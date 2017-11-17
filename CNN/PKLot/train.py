import tensorflow as tf
from CNN.PKLot.model import *

from CNN.PKLot.input_data import *


def run_training():

    N_CLASSES = 2
    BATCH_SIZE = 128
    MAX_STEP = 2000
    learning_rate = 0.0001
    IMG_W = 28
    IMG_H = 28
    CHANNEL = 1
    is_train = 2

    tfrecords_path = '/home/liming/Documents/datasets/PKLot/tfrecords'
    train_tfrecords_name = 'train'
    val_tfrecords_name = 'val'
    logs_train_dir = './logs/train/'
    logs_val_dir = './logs/val/'
    img_path = '/home/liming/Documents/datasets/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_07_34_01.jpg'
    xml_path = '/home/liming/Documents/datasets/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_07_34_01.xml'


    # read data from tfrecords file
    train_batch, train_label_batch = read_and_decode(
                            os.path.join(tfrecords_path, train_tfrecords_name + '.tfrecords'), BATCH_SIZE)
    val_batch, val_label_batch = read_and_decode(os.path.join(tfrecords_path, train_tfrecords_name + '.tfrecords'),
                                                 BATCH_SIZE)

    train_logits = inference(train_batch, BATCH_SIZE, N_CLASSES)

    #####
    train_logits = tf.nn.softmax(train_logits)

    train_loss = losses(train_logits, train_label_batch)
    train_op = training(train_loss, learning_rate)
    train_acc = evaluation(train_logits, train_label_batch)

    x = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_H, CHANNEL])
    y_ = tf.placeholder(tf.int16, shape=[None])

    # save the model, only keep one version
    # create network or initialize network before initilizing the Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            if is_train==1:

                max_acc = 0

                # starts to train
                for step in np.arange(MAX_STEP):
                    if coord.should_stop():
                        break
                    train_images, train_labels = sess.run([train_batch, train_label_batch])
                    _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc],
                                                    feed_dict={x:train_images, y_:train_labels})

                    if step % 50 == 0:
                        print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                        summary_str = sess.run(summary_op)
                        train_writer.add_summary(summary_str, step)

                    if step % 200 == 0 or (step + 1) == MAX_STEP:
                        val_images, val_labels = sess.run([val_batch, val_label_batch])
                        val_loss, val_acc = sess.run([train_loss, train_acc], feed_dict={x:val_images, y_:val_labels})
                        print('** Step %d, val loss = %.2f, val accuracy = %.2f%% **' %(step, val_loss, val_acc*100.0))
                        summary_str = sess.run(summary_op)
                        val_writer.add_summary(summary_str, step)

                        # save the model, save the highest accurate generation
                        # if val_acc > max_acc:
                        #     max_acc = val_acc
                        #     checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                        #     saver.save(sess,checkpoint_path, global_step=step)

                    if step % 200 == 0 or (step + 1) == MAX_STEP:
                        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
            else:
                print("Reading checkpoints...")
                ckpt = tf.train.get_checkpoint_state(logs_train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Loading success, global_step is %s' % global_step)
                else:
                    print('No checkpoint file found')

                train_images, train_labels = sess.run([train_batch, train_label_batch])
                prediction = sess.run(train_logits, feed_dict={x: train_images})
                max_index = np.argmax(prediction, 1)
                print('prediction:', max_index)
                print('label:', train_labels)
                print()
        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.request_stop()

        coord.join(threads)

if __name__ == '__main__':
    run_training()