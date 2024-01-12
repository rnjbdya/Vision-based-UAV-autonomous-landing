def SoundBuzzer(turnoff):
    # Set the GPIO pin number
    buzzer_pin = 15

    # Set the PWM frequency and duty cycle
    frequency = 2000  # Frequency in Hz
    duty_cycle = 50  # Duty cycle in percentage
    try:
        import Jetson.GPIO as GPIO
        # Configure the GPIO mode
        GPIO.setmode(GPIO.BOARD)

        # Set up the GPIO pin as a PWM output
        GPIO.setup(buzzer_pin, GPIO.OUT)
        buzzer_pwm = GPIO.PWM(buzzer_pin, frequency)

        # Start the PWM output
        buzzer_pwm.start(duty_cycle)

        turnoff.wait()

        # Stop the PWM output and clean up
        buzzer_pwm.stop()
        GPIO.cleanup()
    except:
        print('Not running on jetson. Buzzer is on.')
        turnoff.wait()
        print('Buzzer turned off')
