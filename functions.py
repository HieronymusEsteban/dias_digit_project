def complete_image_ids(image_id_strings):
    new_strings = []
    for string in image_id_strings:
        if len(string) == 3:
            new_string = 'id' + string
            new_strings.append(new_string)
        elif len(string) == 2:
            new_string = 'id0' + string
            new_strings.append(new_string)
        elif len(string) == 1:
            new_string = 'id00' + string
            new_strings.append(new_string)
    return new_strings


def reconvert_image_ids(image_ids):
    new_image_ids = []
    for image_id in image_ids:
        new_image_id = image_id.split('id')[1]
        new_image_ids.append(new_image_id)
    return new_image_ids