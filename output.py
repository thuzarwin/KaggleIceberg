# -*- coding: utf-8 -*-


def output(filename, result):
    """
    Output the result to file
    :param filename: xxx.csv
    :param result: [(id, prob)]
    :return: None
    """

    f = open(filename, 'w+')
    line = 'id,is_iceberg\n'
    f.write(line)
    for (_id, prob) in result:
        line = '%s,%s\n' % (_id, prob)
        f.write(line)

    f.close()
    print("Finish Write File: %s" % filename)