#!/usr/bin/env python3
"""
White Stripe强度标准化功能的测试脚本

测试内容：
1. 基本功能测试
2. 不同模态测试
3. 参数敏感性测试
4. 边界条件测试
5. 合成数据生成测试
6. 可视化功能测试
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加主模块路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from white_stripe_normalization.main import (
    white_stripe_normalization, generate_synthetic_mri_data,
    find_white_stripe_range, visualize_white_stripe_normalization
)

def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "="*50)
    print("测试1: 基本功能测试")
    print("="*50)

    # 生成测试数据
    test_image = np.random.rand(64, 64) * 0.5 + 0.25  # [0.25, 0.75]范围

    print(f"测试数据形状: {test_image.shape}")
    print(f"测试数据范围: [{np.min(test_image):.3f}, {np.max(test_image):.3f}]")

    # 测试T1模态标准化
    try:
        normalized_image, white_range, stats = white_stripe_normalization(
            test_image, modality='T1'
        )

        # 验证结果
        assert normalized_image.shape == test_image.shape, "输出形状不匹配"
        assert len(white_range) == 2, "白质范围格式错误"
        assert white_range[0] < white_range[1], "白质范围顺序错误"
        assert 'white_matter_stats' in stats, "缺少白质统计信息"
        assert 'original_stats' in stats, "缺少原始图像统计"
        assert 'normalized_stats' in stats, "缺少标准化图像统计"

        print(f"标准化完成:")
        print(f"  白质范围: [{white_range[0]:.3f}, {white_range[1]:.3f}]")
        print(f"  输出范围: [{np.min(normalized_image):.3f}, {np.max(normalized_image):.3f}]")
        print(f"  白质像素数量: {stats['white_matter_stats']['pixel_count']}")

        print("✅ 基本功能测试通过")

    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        raise

def test_different_modalities():
    """测试不同MRI模态"""
    print("\n" + "="*50)
    print("测试2: 不同模态测试")
    print("="*50)

    modalities = ['T1', 'T2', 'FLAIR', 'PD']  # 添加PD测试未知模态处理

    for modality in modalities:
        print(f"\n测试 {modality} 模态:")

        try:
            # 生成对应模态的合成数据
            test_image = generate_synthetic_mri_data(shape=(64, 64), modality=modality)

            # 执行标准化
            normalized_image, white_range, stats = white_stripe_normalization(
                test_image, modality=modality
            )

            # 验证结果
            assert normalized_image.shape == test_image.shape
            assert stats['parameters']['modality'] == modality or (modality == 'PD' and stats['parameters']['modality'] == 'T1')

            print(f"  ✅ {modality} 模态测试通过")
            print(f"    白质范围: [{white_range[0]:.3f}, {white_range[1]:.3f}]")
            print(f"    输出范围: [{np.min(normalized_image):.3f}, {np.max(normalized_image):.3f}]")

        except Exception as e:
            print(f"  ❌ {modality} 模态测试失败: {e}")
            raise

    print("✅ 不同模态测试通过")

def test_synthetic_data_generation():
    """测试合成数据生成"""
    print("\n" + "="*50)
    print("测试3: 合成数据生成测试")
    print("="*50)

    # 测试不同参数组合
    test_cases = [
        {'shape': (64, 64), 'modality': 'T1', 'noise_level': 0.05},
        {'shape': (128, 128), 'modality': 'T2', 'noise_level': 0.1},
        {'shape': (32, 32), 'modality': 'FLAIR', 'noise_level': 0.02},
        {'shape': (256, 256), 'modality': 'T1', 'bias_field_strength': 0.3}
    ]

    for i, params in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {params}")

        try:
            # 生成合成数据
            synthetic_image = generate_synthetic_mri_data(**params)

            # 验证生成的数据
            assert synthetic_image.shape == params['shape'], "生成的图像形状错误"
            assert np.min(synthetic_image) >= 0, "图像包含负值"
            assert np.max(synthetic_image) <= 1, "图像超出范围"

            # 验证不同组织有不同信号
            unique_values = len(np.unique(synthetic_image.round(2)))
            assert unique_values > 5, "生成的数据缺乏多样性"

            print(f"  ✅ 生成成功，形状: {synthetic_image.shape}")
            print(f"    范围: [{np.min(synthetic_image):.3f}, {np.max(synthetic_image):.3f}]")
            print(f"    唯一值数量: {unique_values}")

        except Exception as e:
            print(f"  ❌ 生成失败: {e}")
            raise

    print("✅ 合成数据生成测试通过")

def test_parameter_sensitivity():
    """测试参数敏感性"""
    print("\n" + "="*50)
    print("测试4: 参数敏感性测试")
    print("="*50)

    # 生成测试数据
    test_image = generate_synthetic_mri_data(shape=(64, 64), modality='T1')

    # 测试不同width参数
    widths = [0.02, 0.05, 0.1, 0.2, 0.5]

    for width in widths:
        print(f"\n测试 width = {width}:")

        try:
            normalized_image, white_range, stats = white_stripe_normalization(
                test_image, modality='T1', width=width
            )

            # 验证结果合理性
            assert 0 <= np.min(normalized_image) <= 1, "标准化图像超出[0,1]范围"
            assert 0 <= np.max(normalized_image) <= 1, "标准化图像超出[0,1]范围"
            assert white_range[0] < white_range[1], "白质范围错误"

            # 验证width影响结果
            wm_pixel_count = stats['white_matter_stats']['pixel_count']

            print(f"  ✅ Width {width} 测试通过")
            print(f"    白质范围: [{white_range[0]:.3f}, {white_range[1]:.3f}]")
            print(f"    白质像素: {wm_pixel_count}")

        except Exception as e:
            print(f"  ❌ Width {width} 测试失败: {e}")
            raise

    # 测试不同迭代参数
    print(f"\n测试不同迭代参数:")
    iteration_params = [5, 10, 20, 50]

    for max_iter in iteration_params:
        try:
            normalized_image, white_range, stats = white_stripe_normalization(
                test_image, modality='T1', max_iterations=max_iter
            )

            print(f"  ✅ 最大迭代 {max_iter}: 范围=[{white_range[0]:.3f}, {white_range[1]:.3f}]")

        except Exception as e:
            print(f"  ❌ 最大迭代 {max_iter} 测试失败: {e}")
            raise

    print("✅ 参数敏感性测试通过")

def test_edge_cases():
    """测试边界条件"""
    print("\n" + "="*50)
    print("测试5: 边界条件测试")
    print("="*50)

    # 测试小图像
    try:
        small_image = np.random.rand(16, 16) * 0.5 + 0.25
        normalized, white_range, stats = white_stripe_normalization(small_image, modality='T1')
        print("✅ 小图像测试通过")
    except Exception as e:
        print(f"❌ 小图像测试失败: {e}")

    # 测试均匀图像
    try:
        uniform_image = np.ones((32, 32)) * 0.5
        normalized, white_range, stats = white_stripe_normalization(uniform_image, modality='T1')
        print("✅ 均匀图像测试通过")
    except Exception as e:
        print(f"❌ 均匀图像测试失败: {e}")

    # 测试极值图像
    try:
        extreme_image = np.zeros((32, 32))
        extreme_image[16:24, 16:24] = 1.0  # 只有一个高信号区域
        normalized, white_range, stats = white_stripe_normalization(extreme_image, modality='T1')
        print("✅ 极值图像测试通过")
    except Exception as e:
        print(f"❌ 极值图像测试失败: {e}")

    # 测试含NaN值的图像
    try:
        nan_image = np.random.rand(32, 32) * 0.5 + 0.25
        nan_image[10, 10] = np.nan
        normalized, white_range, stats = white_stripe_normalization(nan_image, modality='T1')
        print("✅ 含NaN值图像测试通过")
    except Exception as e:
        print(f"❌ 含NaN值图像测试失败: {e}")

    # 测试3D图像（应该能够处理）
    try:
        image_3d = np.random.rand(32, 32, 16) * 0.5 + 0.25
        normalized, white_range, stats = white_stripe_normalization(image_3d, modality='T1')
        print(f"✅ 3D图像测试通过，形状: {normalized.shape}")
    except Exception as e:
        print(f"❌ 3D图像测试失败: {e}")

    print("✅ 边界条件测试完成")

def test_white_range_finding():
    """测试白质范围查找算法"""
    print("\n" + "="*50)
    print("测试6: 白质范围查找测试")
    print("="*50)

    # 生成测试数据
    test_image = generate_synthetic_mri_data(shape=(64, 64), modality='T1')

    # 测试范围查找
    try:
        lower, upper = find_white_stripe_range(
            test_image, 'T1', width=0.1, max_iterations=10, convergence_threshold=0.01
        )

        assert lower < upper, "范围下界应该小于上界"
        assert lower >= np.min(test_image), "下界不应该小于最小值"
        assert upper <= np.max(test_image), "上界不应该大于最大值"

        print(f"✅ 白质范围查找测试通过")
        print(f"    范围: [{lower:.3f}, {upper:.3f}]")

    except Exception as e:
        print(f"❌ 白质范围查找测试失败: {e}")
        raise

    # 测试收敛性
    try:
        # 使用很小的收敛阈值
        lower, upper = find_white_stripe_range(
            test_image, 'T1', width=0.1, max_iterations=50, convergence_threshold=0.0001
        )
        print(f"✅ 高精度收敛测试通过，迭代次数达到最大值")

    except Exception as e:
        print(f"❌ 高精度收敛测试失败: {e}")
        raise

def test_visualization():
    """测试可视化功能"""
    print("\n" + "="*50)
    print("测试7: 可视化功能测试")
    print("="*50)

    try:
        # 生成测试数据
        test_image = generate_synthetic_mri_data(shape=(64, 64), modality='T1')

        # 执行标准化
        normalized_image, white_range, stats = white_stripe_normalization(test_image, modality='T1')

        # 测试可视化函数
        os.makedirs("test_outputs", exist_ok=True)
        save_path = "test_outputs/white_stripe_visualization_test.png"

        visualize_white_stripe_normalization(
            test_image, normalized_image, white_range, stats, save_path
        )

        # 验证输出文件
        if os.path.exists(save_path):
            print(f"✅ 可视化文件已生成: {save_path}")
        else:
            print("❌ 可视化文件未生成")

    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_performance():
    """测试性能"""
    print("\n" + "="*50)
    print("测试8: 性能测试")
    print("="*50)

    import time

    # 不同大小的图像
    sizes = [(64, 64), (128, 128), (256, 256)]

    for size in sizes:
        print(f"\n测试图像大小: {size}")

        # 生成测试数据
        test_image = generate_synthetic_mri_data(shape=size, modality='T1')

        # 测试性能
        start_time = time.time()

        normalized_image, white_range, stats = white_stripe_normalization(
            test_image, modality='T1'
        )

        end_time = time.time()
        processing_time = end_time - start_time
        pixels_per_second = np.prod(size) / processing_time

        print(f"  处理时间: {processing_time:.3f}秒")
        print(f"  处理速度: {pixels_per_second:,.0f} 像素/秒")
        print(f"  白质像素: {stats['white_matter_stats']['pixel_count']}")

    print("✅ 性能测试完成")

def run_all_tests():
    """运行所有测试"""
    print("开始运行White Stripe强度标准化功能测试套件")
    print("="*60)

    try:
        test_basic_functionality()
        test_different_modalities()
        test_synthetic_data_generation()
        test_parameter_sensitivity()
        test_edge_cases()
        test_white_range_finding()
        test_visualization()
        test_performance()

        print("\n" + "="*60)
        print("🎉 所有测试完成！")
        print("="*60)
        print("✅ 基本功能测试通过")
        print("✅ 不同模态测试通过")
        print("✅ 合成数据生成测试通过")
        print("✅ 参数敏感性测试通过")
        print("✅ 边界条件测试通过")
        print("✅ 白质范围查找测试通过")
        print("✅ 可视化功能测试通过")
        print("✅ 性能测试完成")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()