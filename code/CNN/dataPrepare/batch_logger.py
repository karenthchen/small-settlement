
# #in main code:
# batch_logger = NBatchLogger(log_step_freq, out_path, model_nr, initial_epoch)
# callbacks_list = [checkpoint, batch_logger] #for example
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

class NBatchLogger(Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display, out_path, model_nr, initial_epoch):

        #naming of output file
        self.model_nr = model_nr
        self.out_path = out_path
        self.step_info_filename = self.out_path + self.model_nr + '_steplossesinfo.txt'

        self.display = display #saving frequency in steps
        self.epochnr = initial_epoch

        open(self.step_info_filename, 'w').close()
        self.step = 0
        self.metric_cache = {}
        ##self.metrics = {'loss', 'acc'}


    def on_epoch_end(self, epoch, logs=None):
        self.epochnr += 1

    def on_batch_end(self, batch, logs={}):
        self.step += 1

        #print(self.params['metrics'])
        #print(logs)

        # input = self.generator[0]
        # y_predict = np.asarray(self.model.predict(input))
        # _data = []
        # _data.append({'l1': y_predict[0], 'l2': y_predict[2]})
        # print(_data)


        #store loss values
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k] #accumulate new loss

        #print(self.metric_cache.items())
        #print(self.metric_cache)

        #write loss values
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display # compute average for steps accumulated since last writing
                if abs(val) > 1e-3:
                    metrics_log += ' %.4f' % (val)
                else:
                    metrics_log += ' %.4e' % (val)

            current_lr = K.get_value(self.model.optimizer.lr)


            #loss, loss1, loss2, precision1, precision2 = self.model.get_layer('custom_multi_loss_layer_1').output

            #add new line in losses file
            with open(self.step_info_filename, 'a') as self.step_info_file:
                self.step_info_file.write('{} {} {} {} {} \n'.format(self.step,
                                          self.params['steps'], self.epochnr,
                                          metrics_log, current_lr))
            #restart accumulating losses
            self.metric_cache.clear()
