#!/usr/bin/env python3
"""
调试窗口检测功能

Author: RainbowBird
"""

try:
    import Quartz
    
    def debug_window_detection():
        """调试窗口检测"""
        print("🔍 调试窗口检测功能...")
        
        # 获取所有窗口信息
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )
        
        balatro_keywords = ['Balatro']
        excluded_apps = [
            'Cursor', 'Visual Studio Code', 'Code', 'Xcode', 
            'Terminal', 'iTerm', 'Finder', 'Safari', 'Chrome',
            'Firefox', 'TextEdit', 'Sublime Text', 'Atom'
        ]
        
        print(f"\n📋 总共找到 {len(window_list)} 个窗口")
        print("🔍 检查小丑牌窗口...")
        
        candidates = []
        
        for i, window in enumerate(window_list):
            window_name = window.get('kCGWindowName', '')
            window_owner = window.get('kCGWindowOwnerName', '')
            bounds = window.get('kCGWindowBounds', {})
            
            # 只检查有名称的窗口
            if not window_name and not window_owner:
                continue
                
            print(f"  {i+1}. 检查窗口: '{window_name}' | '{window_owner}' | 尺寸: {bounds.get('Width', 0)}x{bounds.get('Height', 0)}")
            
            # 跳过没有尺寸信息的窗口
            if not bounds or bounds.get('Width', 0) < 100 or bounds.get('Height', 0) < 100:
                print(f"     ❌ 跳过：尺寸太小")
                continue
            
            # 排除明确不是游戏的应用
            is_excluded = any(excluded_app.lower() in window_owner.lower() for excluded_app in excluded_apps)
            if is_excluded:
                print(f"     ❌ 跳过：在排除列表中")
                continue
            
            # 检查是否匹配小丑牌关键词
            for keyword in balatro_keywords:
                if (keyword.lower() in window_name.lower() or 
                    keyword.lower() in window_owner.lower()):
                    
                    print(f"     ✅ 匹配关键词: {keyword}")
                    candidates.append({
                        'name': window_name,
                        'owner': window_owner,
                        'bounds': bounds
                    })
                    break
            else:
                print(f"     ⚪ 不匹配关键词")
        
        print(f"\n🎯 找到 {len(candidates)} 个候选窗口:")
        for i, candidate in enumerate(candidates):
            print(f"  {i+1}. '{candidate['name']}' | '{candidate['owner']}' | {candidate['bounds'].get('Width', 0)}x{candidate['bounds'].get('Height', 0)}")
        
        return candidates

    if __name__ == "__main__":
        debug_window_detection()
        
except ImportError:
    print("❌ pyobjc 不可用")
