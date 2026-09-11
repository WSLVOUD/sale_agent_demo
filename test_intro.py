"""
测试动态生成的自我介绍、过渡语和名片话术
"""
from src.first_contact.handler import first_contact_handler

def test_dynamic_intro():
    print("=" * 80)
    print("Test: First Contact Flow with Dynamic Intro, Transition, and Business Card")
    print("=" * 80)
    
    # 模拟三个不同客户的首次消息
    test_cases = [
        ("session_001", "I need a display for my conference room"),
        ("session_002", "Looking for outdoor LED screens for rental"),
    ]
    
    for session_id, customer_msg in test_cases:
        print(f"\n[Customer Message]: {customer_msg}")
        print("-" * 80)
        
        result = first_contact_handler.run(session_id, customer_msg, language="en")
        
        print(f"[1. Self Introduction]:\n{result.intro_text}\n")
        
        print(f"[2. Assets Sent]: {len([r for r in result.asset_results if r.success])} successful")
        for r in result.asset_results:
            if r.success:
                print(f"   - {r.asset.asset_type}: {r.asset.name}")
        
        print(f"\n[3. Transition Message]:\n{result.transition_text}\n")
        
        print(f"[4. Business Card]:\n   URL: {result.business_card_url}\n   Message: {result.business_card_text}\n")
        
        print(f"[Status]: Intro={'OK' if result.intro_success else 'FAIL'}, "
              f"Assets={'OK' if result.all_success else 'PARTIAL'}, "
              f"Transition={'OK' if result.transition_success else 'FAIL'}, "
              f"Card={'OK' if result.business_card_success else 'FAIL'}")
        print("=" * 80)

if __name__ == "__main__":
    test_dynamic_intro()
