# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import subprocess
import pytest


@pytest.mark.light
def test_cqd():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    cmd_str = 'python3 cli/cqda-cli.py --do_test --data_path data/NELL-betae-tiny -n 1 -b 1000 -d 1000 -lr 0.1 ' \
              '--max_steps 1000 --cpu_num 0 --geo cqd --valid_steps 20 ' \
              '--tasks 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up --print_on_screen --test_batch_size 1 ' \
              '--optimizer adagrad --reg_weight 0.05 --log_steps 5 --checkpoint_path models/nell-betae ' \
              '--cqd discrete --cqd-t-norm prod --cqd-k 4'

    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = p.communicate()

    sys.stdout = sys.stderr

    lines = out.decode("utf-8").split("\n")

    sanity_check_flag_1 = False
    sanity_check_flag_2 = False
    sanity_check_flag_3 = False
    sanity_check_flag_4 = False
    sanity_check_flag_5 = False
    sanity_check_flag_6 = False
    sanity_check_flag_7 = False
    sanity_check_flag_8 = False
    sanity_check_flag_9 = False
    sanity_check_flag_10 = False
    sanity_check_flag_11 = False
    sanity_check_flag_12 = False
    sanity_check_flag_13 = False
    sanity_check_flag_14 = False

    for line in lines:
        print(line)

        if 'Test 1p MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.576179, atol=1e-4, rtol=1e-4)
            sanity_check_flag_1 = True
        elif 'Test 2p MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.217309, atol=1e-4, rtol=1e-4)
            sanity_check_flag_2 = True
        elif 'Test 3p MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.138355, atol=1e-4, rtol=1e-4)
            sanity_check_flag_3 = True
        elif 'Test 2i MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.448377, atol=1e-4, rtol=1e-4)
            sanity_check_flag_4 = True
        elif 'Test 3i MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.472417, atol=1e-4, rtol=1e-4)
            sanity_check_flag_5 = True
        elif 'Test ip MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.266512, atol=1e-4, rtol=1e-4)
            sanity_check_flag_6 = True
        elif 'Test pi MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.225731, atol=1e-4, rtol=1e-4)
            sanity_check_flag_7 = True
        elif 'Test 2in MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.003777, atol=1e-4, rtol=1e-4)
            sanity_check_flag_8 = True
        elif 'Test 3in MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.000139, atol=1e-4, rtol=1e-4)
            sanity_check_flag_9 = True
        elif 'Test inp MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.036229, atol=1e-4, rtol=1e-4)
            sanity_check_flag_10 = True
        elif 'Test pin MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.003973, atol=1e-4, rtol=1e-4)
            sanity_check_flag_11 = True
        elif 'Test pni MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.024000, atol=1e-4, rtol=1e-4)
            sanity_check_flag_12 = True
        elif 'Test 2u-DNF MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.014596, atol=1e-4, rtol=1e-4)
            sanity_check_flag_13 = True
        elif 'Test up-DNF MRR at step 99999' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.018403, atol=1e-4, rtol=1e-4)
            sanity_check_flag_14 = True

    assert sanity_check_flag_1
    assert sanity_check_flag_2
    assert sanity_check_flag_3
    assert sanity_check_flag_4
    assert sanity_check_flag_5
    assert sanity_check_flag_6
    assert sanity_check_flag_7
    assert sanity_check_flag_8
    assert sanity_check_flag_9
    assert sanity_check_flag_10
    assert sanity_check_flag_11
    assert sanity_check_flag_12
    assert sanity_check_flag_13
    assert sanity_check_flag_14


@pytest.mark.light
def test_train_cqd():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    cmd_str = 'python3 cli/cqda-cli.py --do_train --do_valid --do_test --data_path data/NELL-betae-tiny -n 1 -b 100 -d 200 ' \
              '-lr 0.1 --warm_up_steps 0 --max_steps 10 --cpu_num 0 --geo cqd --valid_steps 500 --tasks 1p ' \
              '--print_on_screen --test_batch_size 1000 --optimizer adagrad --reg_weight 0.1 --log_steps 1 ' \
              '--use-qa-iterator --disable-saving'

    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = p.communicate()

    sys.stdout = sys.stderr

    lines = out.decode("utf-8").split("\n")

    sanity_check_flag_1 = False
    sanity_check_flag_2 = False
    sanity_check_flag_3 = False
    sanity_check_flag_4 = False
    sanity_check_flag_5 = False
    sanity_check_flag_6 = False
    sanity_check_flag_7 = False
    sanity_check_flag_8 = False
    sanity_check_flag_9 = False
    sanity_check_flag_10 = False
    sanity_check_flag_11 = False
    sanity_check_flag_12 = False

    for line in lines:
        print(line)

        if 'Training average loss at step 0:' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 11.056605, atol=1e-4, rtol=1e-4)
            sanity_check_flag_1 = True
        elif 'Training average loss at step 1:' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 11.056726, atol=1e-4, rtol=1e-4)
            sanity_check_flag_2 = True
        elif 'Training average loss at step 2:' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 11.056125, atol=1e-4, rtol=1e-4)
            sanity_check_flag_3 = True
        elif 'Training average loss at step 3' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 11.056038, atol=1e-4, rtol=1e-4)
            sanity_check_flag_4 = True
        elif 'Training average loss at step 4' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 11.056405, atol=1e-4, rtol=1e-4)
            sanity_check_flag_5 = True
        elif 'Training average loss at step 5' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 11.056239, atol=1e-4, rtol=1e-4)
            sanity_check_flag_6 = True
        elif 'Training average loss at step 6' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 11.056474, atol=1e-4, rtol=1e-4)
            sanity_check_flag_7 = True
        elif 'Training average loss at step 7' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 11.056270, atol=1e-4, rtol=1e-4)
            sanity_check_flag_8 = True
        elif 'Training average loss at step 8' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 11.056221, atol=1e-4, rtol=1e-4)
            sanity_check_flag_9 = True
        elif 'Training average loss at step 9' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 11.055962, atol=1e-4, rtol=1e-4)
            sanity_check_flag_10 = True
        elif 'Valid average MRR at step 9:' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.022252, atol=1e-4, rtol=1e-4)
            sanity_check_flag_11 = True
        elif 'Test average MRR at step 9:' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.040159, atol=1e-4, rtol=1e-4)
            sanity_check_flag_12 = True

    assert sanity_check_flag_1
    assert sanity_check_flag_2
    assert sanity_check_flag_3
    assert sanity_check_flag_4
    assert sanity_check_flag_5
    assert sanity_check_flag_6
    assert sanity_check_flag_7
    assert sanity_check_flag_8
    assert sanity_check_flag_9
    assert sanity_check_flag_10
    assert sanity_check_flag_11
    assert sanity_check_flag_12


@pytest.mark.light
def test_train_cqd_no_warmup():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    cmd_str = 'python3 cli/cqda-cli.py --do_train --do_valid --do_test --data_path data/NELL-betae-tiny -n 1 -b 100 -d 200 ' \
              '-lr 0.1 --warm_up_steps 0 --max_steps 10 --cpu_num 0 --geo cqd --valid_steps 500 --tasks 1p ' \
              '--print_on_screen --test_batch_size 1000 --optimizer adagrad --reg_weight 0.1 --log_steps 1 ' \
              '--use-qa-iterator --disable-saving --disable-warmup'

    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = p.communicate()

    sys.stdout = sys.stderr

    lines = out.decode("utf-8").split("\n")

    sanity_check_flag_1 = False
    sanity_check_flag_2 = False
    sanity_check_flag_3 = False
    sanity_check_flag_4 = False
    sanity_check_flag_5 = False
    sanity_check_flag_6 = False
    sanity_check_flag_7 = False
    sanity_check_flag_8 = False
    sanity_check_flag_9 = False
    sanity_check_flag_10 = False
    sanity_check_flag_11 = False
    sanity_check_flag_12 = False

    for line in lines:
        print(line)

        if 'Training average loss at step 0:' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 11.056605, atol=1e-4, rtol=1e-4)
            sanity_check_flag_1 = True
        elif 'Training average loss at step 1:' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 11.056726, atol=1e-4, rtol=1e-4)
            sanity_check_flag_2 = True
        elif 'Training average loss at step 2:' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 11.049435, atol=1e-4, rtol=1e-4)
            sanity_check_flag_3 = True
        elif 'Training average loss at step 3' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 10.969775, atol=1e-4, rtol=1e-4)
            sanity_check_flag_4 = True
        elif 'Training average loss at step 4' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 10.966204, atol=1e-4, rtol=1e-4)
            sanity_check_flag_5 = True
        elif 'Training average loss at step 5' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 10.729454, atol=1e-4, rtol=1e-4)
            sanity_check_flag_6 = True
        elif 'Training average loss at step 6' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 10.852414, atol=1e-4, rtol=1e-4)
            sanity_check_flag_7 = True
        elif 'Training average loss at step 7' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 10.922649, atol=1e-4, rtol=1e-4)
            sanity_check_flag_8 = True
        elif 'Training average loss at step 8' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 10.758240, atol=1e-4, rtol=1e-4)
            sanity_check_flag_9 = True
        elif 'Training average loss at step 9' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 10.694625, atol=1e-4, rtol=1e-4)
            sanity_check_flag_10 = True
        elif 'Valid average MRR at step 9:' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.076503, atol=1e-4, rtol=1e-4)
            sanity_check_flag_11 = True
        elif 'Test average MRR at step 9:' in line:
            value = float(line.split()[9])
            np.testing.assert_allclose(value, 0.103323, atol=1e-4, rtol=1e-4)
            sanity_check_flag_12 = True

    assert sanity_check_flag_1
    assert sanity_check_flag_2
    assert sanity_check_flag_3
    assert sanity_check_flag_4
    assert sanity_check_flag_5
    assert sanity_check_flag_6
    assert sanity_check_flag_7
    assert sanity_check_flag_8
    assert sanity_check_flag_9
    assert sanity_check_flag_10
    assert sanity_check_flag_11
    assert sanity_check_flag_12


if __name__ == '__main__':
    pytest.main([__file__])
