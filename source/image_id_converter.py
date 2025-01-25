def complete_image_ids(image_id_strings):
    """The last part of the filename is used as
    image identifier. These are numbers that
    start with zeros, which are identified as
    numbers by Python (duck typing). The zeros
    have to be added in front as well as the 
    character id to force the data type
    string."""
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
    """The characters 'id' have been
    added in front of the image identifier
    before saving data to force the data type string.
    These can characters can be removed
    when loading the data so that the ids
    match the actual tags of the image file names."""
    new_image_ids = []
    for image_id in image_ids:
        new_image_id = image_id.split('id')[1]
        new_image_ids.append(new_image_id)
    return new_image_ids