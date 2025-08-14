#!/usr/bin/env python3
"""
iframe显示问题诊断工具
"""

import os
import glob
import html

def test_iframe_display():
    """测试iframe显示功能"""
    
    # 查找一个HTML文件进行测试
    html_files = glob.glob('own/**/*.html', recursive=True)
    
    if not html_files:
        print("❌ 没有找到HTML文件进行测试")
        return
    
    test_file = html_files[0]
    print(f"🧪 使用文件进行测试: {test_file}")
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📊 文件大小: {len(content)} 字符")
        
        # 创建测试页面
        test_page_content = create_test_page(content, test_file)
        
        # 保存测试页面
        test_file_path = 'iframe_test.html'
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_page_content)
        
        print(f"✅ 测试页面已创建: {test_file_path}")
        print("🌐 请在浏览器中打开此文件查看iframe显示效果")
        print("📱 或启动Flask应用后访问 /iframe_test 路径")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def create_test_page(original_content, source_file):
    """创建iframe测试页面"""
    
    # HTML转义处理
    escaped_content = html.escape(original_content, quote=True)
    
    # 处理反斜杠和特殊字符，避免f-string问题
    safe_content = original_content.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
    
    # 创建测试页面
    test_page = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>iframe显示测试</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .test-section {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .iframe-container {
            border: 2px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            height: 500px;
            position: relative;
        }
        iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
        .info {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .success {
            background: #e8f5e8;
            color: #2e7d2e;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .code-block {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #dee2e6;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover {
            background: #5a67d8;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 iframe显示诊断测试</h1>
        
        <div class="test-section">
            <h2>📋 测试信息</h2>
            <div class="info">
                <p><strong>源文件:</strong> ''' + source_file + '''</p>
                <p><strong>文件大小:</strong> ''' + str(len(original_content)) + ''' 字符</p>
                <p><strong>测试时间:</strong> <span id="test-time"></span></p>
            </div>
        </div>
        
        <div class="test-section">
            <h2>🖼️ iframe显示测试 (srcdoc方式)</h2>
            <div class="info">
                <p>这个iframe使用srcdoc属性，模拟Flask应用的显示方式</p>
                <button onclick="reloadIframe1()">重新加载</button>
                <button onclick="toggleIframe1()">显示/隐藏</button>
            </div>
            <div class="iframe-container" id="iframe-container-1">
                <iframe id="iframe-1" srcdoc="''' + escaped_content + '''"></iframe>
            </div>
            <div id="status-1" class="info">状态: 加载中...</div>
        </div>
        
        <div class="test-section">
            <h2>📄 直接HTML显示测试</h2>
            <div class="info">
                <p>这里直接显示HTML内容，不使用iframe</p>
                <button onclick="toggleDirect()">显示/隐藏</button>
            </div>
            <div id="direct-content" style="border: 2px solid #ddd; padding: 20px; border-radius: 8px; min-height: 300px;">
                ''' + original_content + '''
            </div>
        </div>
        
        <div class="test-section">
            <h2>🔍 内容分析</h2>
            <div class="info">
                <p><strong>检测结果:</strong></p>
                <ul id="analysis-results"></ul>
            </div>
            
            <h3>原始HTML内容预览 (前1000字符):</h3>
            <div class="code-block">''' + html.escape(original_content[:1000]) + '''</div>
        </div>
        
        <div class="test-section">
            <h2>🛠️ 调试信息</h2>
            <div id="debug-info" class="info">
                <p>等待iframe加载...</p>
            </div>
            <button onclick="runDiagnostics()">运行诊断</button>
            <button onclick="checkIframeContent()">检查iframe内容</button>
        </div>
    </div>

    <script>
        // 更新测试时间
        document.getElementById('test-time').textContent = new Date().toLocaleString();
        
        // iframe加载状态监控
        const iframe1 = document.getElementById('iframe-1');
        const status1 = document.getElementById('status-1');
        
        iframe1.onload = function() {
            status1.innerHTML = '<div class="success">✅ iframe加载成功</div>';
            console.log('iframe加载完成');
        };
        
        iframe1.onerror = function(e) {
            status1.innerHTML = '<div class="error">❌ iframe加载失败: ' + e + '</div>';
            console.error('iframe加载失败:', e);
        };
        
        // 重新加载iframe
        function reloadIframe1() {
            iframe1.src = iframe1.src;
            status1.innerHTML = '<div class="info">🔄 重新加载中...</div>';
        }
        
        // 切换iframe显示
        function toggleIframe1() {
            const container = document.getElementById('iframe-container-1');
            container.style.display = container.style.display === 'none' ? 'block' : 'none';
        }
        
        // 切换直接内容显示
        function toggleDirect() {
            const content = document.getElementById('direct-content');
            content.style.display = content.style.display === 'none' ? 'block' : 'none';
        }
        
        // 运行诊断
        function runDiagnostics() {
            const debugInfo = document.getElementById('debug-info');
            let info = '<h4>🔍 诊断结果:</h4>';
            
            // 检查iframe是否存在
            info += '<p>✅ iframe元素存在: ' + (iframe1 ? '是' : '否') + '</p>';
            
            // 检查iframe尺寸
            const rect = iframe1.getBoundingClientRect();
            info += '<p>📐 iframe尺寸: ' + rect.width + ' x ' + rect.height + '</p>';
            
            // 检查是否可见
            info += '<p>👁️ iframe可见: ' + (rect.width > 0 && rect.height > 0 ? '是' : '否') + '</p>';
            
            // 检查srcdoc内容长度
            info += '<p>📄 srcdoc长度: ' + (iframe1.srcdoc ? iframe1.srcdoc.length : 0) + ' 字符</p>';
            
            // 检查控制台错误
            info += '<p>💭 请查看浏览器控制台是否有错误信息</p>';
            
            debugInfo.innerHTML = info;
        }
        
        // 检查iframe内容
        function checkIframeContent() {
            try {
                const iframeDoc = iframe1.contentDocument || iframe1.contentWindow.document;
                if (iframeDoc) {
                    const debugInfo = document.getElementById('debug-info');
                    debugInfo.innerHTML += '<div class="success">✅ 可以访问iframe内容</div>';
                    debugInfo.innerHTML += '<p>iframe标题: ' + (iframeDoc.title || '无标题') + '</p>';
                    debugInfo.innerHTML += '<p>iframe body存在: ' + (iframeDoc.body ? '是' : '否') + '</p>';
                } else {
                    throw new Error('无法访问iframe内容');
                }
            } catch (e) {
                const debugInfo = document.getElementById('debug-info');
                debugInfo.innerHTML += '<div class="error">❌ 无法访问iframe内容: ' + e.message + '</div>';
            }
        }
        
        // 内容分析
        function analyzeContent() {
            const results = document.getElementById('analysis-results');
            const content = `''' + safe_content + '''`;
            
            let analysis = [];
            
            // 检查内容大小
            if (content.length > 50000) {
                analysis.push('⚠️ 内容较大 (' + content.length + ' 字符)，可能影响iframe显示');
            } else {
                analysis.push('✅ 内容大小正常 (' + content.length + ' 字符)');
            }
            
            // 检查特殊字符
            if (content.includes('"')) {
                analysis.push('⚠️ 包含双引号，可能影响srcdoc属性');
            }
            
            // 检查脚本
            if (content.includes('<script')) {
                analysis.push('📜 包含JavaScript脚本');
            }
            
            // 检查图表库
            if (content.includes('Plotly')) {
                analysis.push('📊 使用Plotly图表库');
            }
            if (content.includes('Chart')) {
                analysis.push('📈 使用Chart.js图表库');
            }
            
            // 检查CSS
            if (content.includes('<style') || content.includes('style=')) {
                analysis.push('🎨 包含CSS样式');
            }
            
            results.innerHTML = analysis.map(item => '<li>' + item + '</li>').join('');
        }
        
        // 页面加载完成后运行分析
        window.addEventListener('DOMContentLoaded', function() {
            setTimeout(analyzeContent, 1000);
            setTimeout(runDiagnostics, 2000);
        });
        
        // 全局错误处理
        window.addEventListener('error', function(e) {
            console.error('页面错误:', e.message);
        });
    </script>
</body>
</html>'''
    
    return test_page

def add_test_route_to_flask():
    """为Flask应用添加测试路由"""
    route_code = '''
@app.route('/iframe_test')
def iframe_test():
    """iframe显示测试页面"""
    html_files = glob.glob('own/**/*.html', recursive=True)
    
    if not html_files:
        return "<h1>没有找到HTML文件进行测试</h1>"
    
    test_file = html_files[0]
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建测试页面
        from iframe_debug import create_test_page
        test_page = create_test_page(content, test_file)
        
        return test_page
        
    except Exception as e:
        return f"<h1>测试失败: {e}</h1>"
'''
    
    print("📋 将以下代码添加到您的Flask应用中:")
    print("=" * 50)
    print(route_code)
    print("=" * 50)

def main():
    """主函数"""
    print("🔍 iframe显示问题诊断工具")
    print("=" * 50)
    
    if len(os.sys.argv) > 1:
        if os.sys.argv[1] == 'flask':
            add_test_route_to_flask()
            return
    
    print("🧪 创建iframe显示测试...")
    test_iframe_display()
    
    print("\n💡 使用说明:")
    print("   1. 打开生成的 iframe_test.html 文件")
    print("   2. 观察iframe是否正常显示内容")
    print("   3. 对比直接HTML显示的效果")
    print("   4. 查看诊断信息和分析结果")
    print("   5. 运行 'python iframe_debug.py flask' 获取Flask集成代码")

if __name__ == '__main__':
    main()