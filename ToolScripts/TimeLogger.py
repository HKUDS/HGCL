import datetime

logmsg = ''
timemark = dict()
saveDefault = False
def log(msg, save=None, oneline=False):
	global logmsg
	global saveDefault
	time = datetime.datetime.now()
	tem = '%s: %s' % (time, msg)
	if save != None:
		if save:
			logmsg += tem + '\n'
	elif saveDefault:
		logmsg += tem + '\n'
	if oneline:
		print(tem, end='\r')
	else:
		print(tem)

def marktime(marker):
	global timemark
	timemark[marker] = datetime.datetime.now()

def SpentTime(marker):
	global timemark
	if marker not in timemark:
		msg = 'LOGGER ERROR, marker', marker, ' not found'
		tem = '%s: %s' % (time, msg)
		print(tem)
		return False
	return datetime.datetime.now() - timemark[marker]

def SpentTooLong(marker, day=0, hour=0, minute=0, second=0):
	global timemark
	if marker not in timemark:
		msg = 'LOGGER ERROR, marker', marker, ' not found'
		tem = '%s: %s' % (time, msg)
		print(tem)
		return False
	return datetime.datetime.now() - timemark[marker] >= datetime.timedelta(days=day, hours=hour, minutes=minute, seconds=second)

if __name__ == '__main__':
	log('')