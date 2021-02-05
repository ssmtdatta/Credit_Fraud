import argparse

def sq(x):
	return x**2


if __name__ == '__main__':

	ap = argparse.ArgumentParser()
	ap.add_argument("-num", "--num", required=True, help="number to square")


	args = vars(ap.parse_args())
	x = int(args['num'])

	x_sq = sq(x)

	print(f"square of {x} is {x_sq}")


	

