


def apply_mult_transform_img(img, transforms_list):
	# APPLY IMAGE TRANSFORMS IN THE ORDER GIVEN IN THE LIST
	# ORDER MATTERS!!
	final_val = None
	for transform in transforms_list:
		final_val = apply_transform_img(img, transform)
	return final_val


def apply_transform_img(img, transform):
	return transform(img)