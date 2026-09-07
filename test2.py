from smbus2 import SMBus, i2c_msg

bus = SMBus(1)

msg = i2c_msg.write(0x70, [0x00, 0x01])
bus.i2c_rdwr(msg)

msg = i2c_msg.write(0x70, [0xE8, 0xAA])
bus.i2c_rdwr(msg)

msg = i2c_msg.write(0x70, [0x02, 255])
bus.i2c_rdwr(msg)

msg = i2c_msg.write(0x70, [0x03, 0])
bus.i2c_rdwr(msg)

msg = i2c_msg.write(0x70, [0x04, 255])
bus.i2c_rdwr(msg)

msg = i2c_msg.write(0x70, [0x05, 0])
bus.i2c_rdwr(msg)