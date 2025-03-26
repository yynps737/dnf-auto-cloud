#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动作生成模块，根据图像识别结果生成游戏操作指令
"""

import logging
import random
import time
from config.settings import DNF, BEHAVIOR

logger = logging.getLogger("DNFAutoCloud")

def generate_actions(detections, game_state):
    """
    根据检测结果和游戏状态生成动作
    
    参数:
        detections (list): 检测结果列表
        game_state (dict): 游戏状态信息
        
    返回:
        list: 操作指令列表
    """
    actions = []
    
    try:
        # 检查是否处于战斗状态
        in_battle = is_in_battle(detections, game_state)
        
        if in_battle:
            # 战斗状态下的操作
            battle_actions = generate_battle_actions(detections, game_state)
            actions.extend(battle_actions)
        else:
            # 非战斗状态下的操作
            navigation_actions = generate_navigation_actions(detections, game_state)
            actions.extend(navigation_actions)
        
        # 根据游戏状态添加额外操作
        additional_actions = generate_additional_actions(detections, game_state)
        actions.extend(additional_actions)
        
        # 添加人类行为特征
        humanize_actions(actions)
        
        return actions
        
    except Exception as e:
        logger.error(f"生成动作时出错: {e}")
        # 出错时返回安全的默认动作（如停止移动）
        return [{"type": "stop", "reason": "error_recovery"}]

def is_in_battle(detections, game_state):
    """
    判断是否处于战斗状态
    
    参数:
        detections (list): 检测结果
        game_state (dict): 游戏状态
        
    返回:
        bool: 是否处于战斗状态
    """
    # 检查是否有怪物或Boss
    for det in detections:
        if det["class_name"] in ["monster", "boss"]:
            return True
    
    # 检查游戏状态中的战斗标志
    return game_state.get("in_battle", False)

def generate_battle_actions(detections, game_state):
    """生成战斗状态下的操作"""
    actions = []
    
    # 获取所有怪物
    monsters = [d for d in detections if d["class_name"] in ["monster", "boss"]]
    
    if monsters:
        # 按照优先级排序（Boss > 普通怪物）
        monsters.sort(key=lambda x: 0 if x["class_name"] == "boss" else 1)
        
        # 选取最近或优先级最高的目标
        target = monsters[0]
        
        # 添加移动到目标附近的动作
        actions.append({
            "type": "move_to",
            "position": target["center"],
            "target_id": target.get("id", "unknown"),
            "target_type": target["class_name"]
        })
        
        # 添加攻击动作
        skill_key = choose_skill(game_state)
        actions.append({
            "type": "use_skill",
            "key": skill_key,
            "target_position": target["center"],
            "skill_name": DNF["skills"].get(skill_key, "未知技能")
        })
    
    return actions

def generate_navigation_actions(detections, game_state):
    """生成非战斗状态下的导航操作"""
    actions = []
    
    # 检查是否有门、NPC等交互物体
    doors = [d for d in detections if d["class_name"] == "door"]
    npcs = [d for d in detections if d["class_name"] == "npc"]
    items = [d for d in detections if d["class_name"] == "item"]
    
    # 优先拾取物品
    if items:
        for item in items:
            actions.append({
                "type": "move_to",
                "position": item["center"],
                "purpose": "pickup_item"
            })
            actions.append({
                "type": "interact",
                "key": "x",  # 假设X键是拾取键
                "purpose": "pickup_item"
            })
    
    # 与NPC交互
    elif npcs and game_state.get("should_talk_to_npc", False):
        npc = npcs[0]
        actions.append({
            "type": "move_to",
            "position": npc["center"],
            "purpose": "talk_to_npc"
        })
        actions.append({
            "type": "interact",
            "key": "f",  # 假设F键是交互键
            "purpose": "talk_to_npc"
        })
    
    # 前往下一个门
    elif doors:
        door = doors[0]
        actions.append({
            "type": "move_to",
            "position": door["center"],
            "purpose": "go_to_next_room"
        })
        actions.append({
            "type": "interact",
            "key": "f",  # 假设F键是交互键
            "purpose": "go_to_next_room"
        })
    
    # 探索地图
    else:
        current_map = game_state.get("current_map", "")
        if current_map in DNF["maps"]:
            # 移动到地图出口
            exit_pos = DNF["maps"][current_map]["exit"]
            actions.append({
                "type": "move_to",
                "position": exit_pos,
                "purpose": "explore_map"
            })
        else:
            # 随机探索
            actions.append({
                "type": "move_random",
                "direction": random.choice(["right", "left", "up", "down"]),
                "duration": random.uniform(0.5, 1.5),
                "purpose": "explore_unknown_map"
            })
    
    return actions

def generate_additional_actions(detections, game_state):
    """生成额外的操作，如使用药水、处理特殊情况等"""
    actions = []
    
    # 检查血量是否过低
    hp_bars = [d for d in detections if d["class_name"] == "hp_bar"]
    if hp_bars:
        hp_bar = hp_bars[0]
        # 假设我们可以从检测结果中获取血量百分比
        hp_percent = estimate_bar_percent(hp_bar["bbox"])
        
        if hp_percent < 30:  # 血量低于30%
            actions.append({
                "type": "use_item",
                "key": "f1",  # 假设F1是使用HP药水的快捷键
                "purpose": "use_hp_potion"
            })
    
    # 检查是否需要使用技能
    # (其他额外操作逻辑...)
    
    return actions

def choose_skill(game_state):
    """选择要使用的技能"""
    # 可用的技能键
    available_skills = list(DNF["skills"].keys())
    
    # 根据游戏状态筛选可用技能
    cooldowns = game_state.get("cooldowns", {})
    available_skills = [s for s in available_skills if s not in cooldowns]
    
    if not available_skills:
        # 如果所有技能都在冷却中，使用普通攻击
        return "1"
    
    # 随机选择一个技能（可以根据策略优化）
    return random.choice(available_skills)

def estimate_bar_percent(bbox):
    """估计血条/蓝条的百分比"""
    # 根据边界框计算百分比
    width = bbox[2] - bbox[0]
    # 假设血条是从左到右填充的
    # 这里的实现取决于具体的血条UI
    return random.uniform(0, 100)  # 临时返回随机值

def humanize_actions(actions):
    """为动作添加人类行为特征"""
    for action in actions:
        # 添加随机延迟
        delay = random.normalvariate(
            (BEHAVIOR["min_delay"] + BEHAVIOR["max_delay"]) / 2,
            (BEHAVIOR["max_delay"] - BEHAVIOR["min_delay"]) / 6
        )
        action["delay"] = max(BEHAVIOR["min_delay"], min(BEHAVIOR["max_delay"], delay))
        
        # 为移动和点击添加随机偏移
        if action["type"] in ["move_to", "use_skill"] and "position" in action:
            x, y = action["position"]
            max_offset_x = BEHAVIOR["click_variance"] * 10
            max_offset_y = BEHAVIOR["click_variance"] * 10
            
            x_offset = random.normalvariate(0, max_offset_x)
            y_offset = random.normalvariate(0, max_offset_y)
            
            action["position"] = [x + x_offset, y + y_offset]
        
        # 添加随机执行顺序标记
        action["execution_priority"] = random.uniform(0.8, 1.2)