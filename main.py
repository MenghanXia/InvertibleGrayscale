import numpy as np
import tensorflow as tf
import datetime, time, scipy.io
from model import *
from util import *

# --------------------------------- HYPER-PARAMETERS --------------------------------- #
in_channels = 3
out_channels = 3
n_epochs1 = 90
n_epochs2 = 30
batch_size = 8
learning_rate = 0.0002
beta1 = 0.9

display_steps = 100
save_epochs = 10
src_suffix = 'target'
dst_suffix = 'target'


def gen_list(data_dir):
    file_list = glob.glob(os.path.join(data_dir, src_suffix, '*.*'))
    file_list.sort()
    file_pair_list = []
    for path1 in file_list:
        path2 = path1.replace(src_suffix, dst_suffix)
        path12 = path1 + ' ' + path2
        file_pair_list.append(path12)
    return file_pair_list


def train(train_list, val_list, debug_mode=True):
    print('Running ColorEncoder -Training!')
    # create folders to save trained model and results
    graph_dir   = './graph'
    checkpt_dir = './checkpoints'
    ouput_dir   = './output'
    exists_or_mkdir(graph_dir, need_remove=True)
    exists_or_mkdir(ouput_dir)
    exists_or_mkdir(checkpt_dir)

    # --------------------------------- load data ---------------------------------
    # data fetched at range: [-1,1]
    input_imgs, target_imgs, num = input_producer(train_list, in_channels, batch_size, need_shuffle=True)

    latent_imgs = encode(input_imgs, 1, is_train=True, reuse=False)
    pred_imgs = decode(latent_imgs, out_channels, is_train=True, reuse=False)
    if debug_mode:
        input_val, target_val, num_val = input_producer(val_list, in_channels, batch_size, need_shuffle=False)
        latent_val = encode(input_val, 1, is_train=False, reuse=True)
        pred_val = decode(latent_val, out_channels, is_train=False, reuse=True)

    # --------------------------------- loss terms ---------------------------------
    with tf.name_scope('Loss'):
        
        target_224 = tf.image.resize_images(target_imgs, size=[224, 224], method=0, align_corners=False)
        predict_224 = tf.image.resize_images(latent_imgs, size=[224, 224], method=0, align_corners=False)
        vgg19_api = VGG19("../vgg19.npy")
        vgg_map_targets = vgg19_api.build((target_224 + 1) / 2, is_rgb=True)
        vgg_map_predict = vgg19_api.build((predict_224 + 1) / 2, is_rgb=False)
        # stretch the global contrast to follow color contrast
        vgg_loss = 1e-7 * tf.losses.mean_squared_error(vgg_map_targets, vgg_map_predict)
        # suppress local patterns
        gray_inputs = tf.image.rgb_to_grayscale(target_imgs)
        latent_grads = tf.reduce_mean(tf.image.total_variation(latent_imgs)/256**2)
        target_grads = tf.reduce_mean(tf.image.total_variation(gray_inputs)/256**2)
        grads_loss = tf.abs(latent_grads-target_grads)
        # control the mapping order similar to normal rgb2gray
        global_order_loss = tf.reduce_mean(tf.maximum(70/127.0, tf.abs(gray_inputs-latent_imgs))) - 70/127.0
        # quantization loss
        latent_stack = tf.concat([latent_imgs for t in range(256)], axis=3)
        id_mat = np.ones(shape=(1, 1, 1, 1))
        quant_stack = np.concatenate([id_mat * t for t in range(256)], axis=3)
        quant_stack = (quant_stack / 127.5) - 1
        quantization_map = tf.reduce_min(tf.abs(latent_stack - quant_stack), axis=3)
        quantization_loss = tf.reduce_mean(quantization_map)

        # reconstruction loss
        mse_loss = tf.losses.mean_squared_error(target_imgs, pred_imgs)
        loss_op1 = 3 * mse_loss + vgg_loss + 0.5*grads_loss + global_order_loss
        loss_op2 = 3 * mse_loss + vgg_loss + 0.1*grads_loss + global_order_loss + 10*quantization_loss

    # --------------------------------- solver definition ---------------------------------
    global_step = tf.Variable(0, name='global_step1', trainable=False)
    iters_per_epoch = np.floor_divide(num, batch_size)
    lr_decay = tf.train.polynomial_decay(learning_rate=learning_rate,
                                          global_step=global_step,
                                          decay_steps=iters_per_epoch*(n_epochs1+n_epochs2),
                                          end_learning_rate=learning_rate / 100.0,
                                          power=0.9)

    with tf.name_scope('optimizer'):
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("encode") or var.name.startswith("decode")]
        train_op1 = tf.train.AdamOptimizer(lr_decay, beta1=beta1).minimize(loss_op1, var_list=gen_vars, global_step=global_step)
        train_op2 = tf.train.AdamOptimizer(lr_decay, beta1=beta1).minimize(loss_op2, var_list=gen_vars, global_step=global_step)

    # --------------------------------- model training ---------------------------------

    with tf.name_scope("parameter_count"):
        num_parameters = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    # set GPU resources
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.45

    saver = tf.train.Saver(max_to_keep=1)
    total_loss_list = []
    grad_loss_list = []
    vgg_loss_list = []
    order_loss_list = []
    quanti_loss_list = []
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        need_resotre = False
        if need_resotre:
            check_pt = tf.train.get_checkpoint_state("checkpoints")
            if check_pt and check_pt.model_checkpoint_path:
                saver.restore(sess, check_pt.model_checkpoint_path)
                print("pretrained model loaded successfully!")

        print(">>------------>>> [Training_Num] =%d" % num)
        print(">>------------>>> [Parameter_Num] =%d" % sess.run(num_parameters))

        # -------------------------------- stage one --------------------------------
        for epoch in range(0, n_epochs1):
            start_time = time.time()
            epoch_loss, n_iters = 0, 0
            avg_grads, avg_vggs, avg_orders = 0, 0, 0
            for step in range(0, num, batch_size):
                _, loss, grads, vggs, orders = sess.run([train_op1, loss_op1, grads_loss, vgg_loss, global_order_loss])
                epoch_loss += loss
                avg_grads += grads
                avg_vggs += vggs
                avg_orders += orders
                n_iters += 1
                # iteration information
                if n_iters % display_steps == 0:
                    tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    print("%s >> [%d/%d] iter: %d  loss: %4.4f" % (tm, epoch, n_epochs1+n_epochs2, n_iters, loss))

            # epoch information
            epoch_loss = epoch_loss / n_iters
            avg_grads = avg_grads / n_iters
            avg_vggs = avg_vggs / n_iters
            avg_orders = avg_orders / n_iters
            total_loss_list.append(epoch_loss)
            grad_loss_list.append(avg_grads)
            vgg_loss_list.append(avg_vggs)
            order_loss_list.append(avg_orders)
            print("[*] ----- Epoch: %d/%d | Loss: %4.4f | Time-consumed: %4.3f -----" %
                  (epoch, n_epochs1+n_epochs2, epoch_loss, (time.time() - start_time)))

            if debug_mode:
                print("----- validating model ...")
                for idx in range(0, num_val, batch_size):
                    latents = sess.run(latent_val)
                    save_images_from_batch(latents, ouput_dir, idx)

            if (epoch+1) % save_epochs == 0:
                print("----- saving model  ...")
                saver.save(sess, os.path.join(checkpt_dir, "model.cpkt"), global_step=global_step)
                save_list(os.path.join(ouput_dir, "total_loss"), total_loss_list)
                save_list(os.path.join(ouput_dir, "grads_loss"), grad_loss_list)
                save_list(os.path.join(ouput_dir, "vggs_loss"), vgg_loss_list)
                save_list(os.path.join(ouput_dir, "order_loss"), order_loss_list)

        # -------------------------------- stage two --------------------------------
        for epoch in range(0, n_epochs2):
            start_time = time.time()
            epoch_loss, n_iters = 0, 0
            avg_grads, avg_vggs, avg_orders, avg_quanti = 0, 0, 0, 0
            for step in range(0, num, batch_size):
                _, loss, grads, vggs, orders, quants = sess.run([train_op2, loss_op2, grads_loss, vgg_loss, global_order_loss, quantization_loss])
                epoch_loss += loss
                avg_grads += grads
                avg_vggs += vggs
                avg_orders += orders
                avg_quanti += quants
                n_iters += 1
                # iteration information
                if n_iters % display_steps == 0:
                    tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    print("%s >> [%d/%d] iter: %d  loss: %4.4f" % (tm, epoch+n_epochs1, n_epochs1+n_epochs2, n_iters, loss))

            # epoch information
            epoch_loss = epoch_loss / n_iters
            avg_grads = avg_grads / n_iters
            avg_vggs = avg_vggs / n_iters
            avg_orders = avg_orders / n_iters
            avg_quanti = avg_quanti / n_iters
            total_loss_list.append(epoch_loss)
            grad_loss_list.append(avg_grads)
            vgg_loss_list.append(avg_vggs)
            order_loss_list.append(avg_orders)
            quanti_loss_list.append(avg_quanti)
            print("[*] ----- Epoch: %d/%d | Loss: %4.4f | Time-consumed: %4.3f -----" %
                  (epoch+n_epochs1, n_epochs1+n_epochs2, epoch_loss, (time.time() - start_time)))

            if debug_mode:
                print("----- validating model ...")
                for idx in range(0, num_val, batch_size):
                    latents = sess.run(latent_val)
                    save_images_from_batch(latents, ouput_dir, idx)

            if (epoch+1) % save_epochs == 0:
                print("----- saving model  ...")
                saver.save(sess, os.path.join(checkpt_dir, "model.cpkt"), global_step=global_step)
                save_list(os.path.join(ouput_dir, "total_loss"), total_loss_list)
                save_list(os.path.join(ouput_dir, "grads_loss"), grad_loss_list)
                save_list(os.path.join(ouput_dir, "vggs_loss"), vgg_loss_list)
                save_list(os.path.join(ouput_dir, "order_loss"), order_loss_list)
                save_list(os.path.join(ouput_dir, "quant_loss"), quanti_loss_list)

        # stop data queue
        coord.request_stop()
        coord.join(threads)
        print ("Training finished!")

    return None


def evaluate(test_list, checkpoint_dir, save_dir_test):
    print('Running ColorEncoder -Evaluation!')
    #save_dir_test = os.path.join("./output/results")
    exists_or_mkdir(save_dir_test)
	
    # ------------- Running Options
    # if run encoder, 3 channel RGB image should be provided in the 'test_list'
	# if run decoder, 1 channel invertible grayscale image should be provided in the 'test_list'
    RUN_Encoder = False

    # --------------------------------- set model ---------------------------------
    # data fetched within range: [-1,1]
    if RUN_Encoder:
	    input_imgs, target_imgs, num = input_producer(test_list, 3, batch_size, need_shuffle=False)
	    latent_imgs = encode(input_imgs, 1, is_train=False, reuse=False)
    else:
	    input_imgs, target_imgs, num = input_producer(test_list, 1, batch_size, need_shuffle=False)
	    restored_imgs = decode(input_imgs, out_channels, is_train=False, reuse=False)

    # --------------------------------- evaluation ---------------------------------
    # set GPU resources
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.45
    num = 10
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Restore model weights from previously saved model
        check_pt = tf.train.get_checkpoint_state(checkpoint_dir)
        if check_pt and check_pt.model_checkpoint_path:
            saver.restore(sess, check_pt.model_checkpoint_path)
            print('model is loaded successfully.')
        else:
            print('# error: loading checkpoint failed.')
            return None

        start_time = time.time()
        print("Total images:%d" % num)
        cnt = 0
        while not coord.should_stop():
            tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            print('%s evaluating: [%d - %d]' % (tm, cnt, cnt+batch_size))
            if RUN_Encoder:			# save the synthesized invertible grayscale
                gray_imgs = sess.run(latent_imgs)
                save_images_from_batch(gray_imgs, save_dir_test, cnt)
            else:							# save the restored color images
                color_imgs = sess.run(restored_imgs)
                save_images_from_batch(color_imgs, save_dir_test, cnt)

            cnt += batch_size
            if cnt >= num:
                coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        print("Testing finished! consumes %f sec" % (time.time() - start_time))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='train, test')
    parser.add_argument('--train_dir', type=str, default='../Dataset/VOC2012/', help='train, test')
    parser.add_argument('--val_dir', type=str, default='../Dataset/color_val/', help='train, test')
    parser.add_argument('--test_dir', type=str, default='./test', help='train, test')
    parser.add_argument('--save_dir', type=str, default='./output', help='train, test')
    args = parser.parse_args()

    if args.mode == 'train':
        train_list = gen_list(args.train_dir)
        val_list = gen_list(args.val_dir)
        train(train_list, val_list, debug_mode=True)
    elif args.mode == 'test':
        test_list = gen_list(args.test_dir)
        checkpoint_dir = "checkpoints"
        evaluate(test_list, checkpoint_dir, args.save_dir)
    else:
        raise Exception("Unknow --mode")
