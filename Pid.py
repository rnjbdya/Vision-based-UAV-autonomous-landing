import time

class pid(object):
    def __init__(self, initial_p=0, initial_i=0, initial_d=0):
        self.p_gain = initial_p
        self.i_gain = initial_i
        self.d_gain = initial_d

        self.last_error = 0
        self.integral_error = 0

        self.last_update = time.time()

    def __str__(self):
        return "P:%s,I:%s,D:%s,Integrator:%s" % (self.p_gain, self.i_gain, self.d_gain, self.integral_error)

    def get_dt(self, max_dt):
        now = time.time()
        time_diff = now - self.last_update
        self.last_update = now
        if time_diff > max_dt:
            return max_dt
        else:
            return time_diff

    def get_p(self, error):
        return self.p_gain * error

    def get_i(self, error, dt):
        self.integral_error = self.integral_error + (error * dt)
        return self.integral_error

    def get_d(self, error, dt):
        if self.last_error is None:
            self.last_error = error
        ret = (error - self.last_error) / (dt + 0.000001)
        self.last_error = error
        return ret

    def get_pi(self, error, dt):
        return (self.get_p(error) * self.p_gain) + (self.get_i(error, dt) * self.i_gain)

    def get_pid(self, error, dt):
        return (self.get_p(error) * self.p_gain) + (self.get_i(error, dt) * self.i_gain) \
            + (self.get_d(error, dt) * self.d_gain)

    def get_integrator(self):
        return self.integral_error * self.i_gain

    def reset_I(self):
        self.integral_error = 0
