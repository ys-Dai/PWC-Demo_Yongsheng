from flask import Flask, render_template, request, jsonify
import os
import glob
import csv
import logging
from werkzeug.utils import secure_filename

import subprocess
import sys
import run  

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['HTML_FOLDER'] = 'own'

# 确保必要目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['HTML_FOLDER'], exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_csv_simple(file_path):
    """简单的CSV解析器，不依赖pandas"""
    try:
        # 尝试不同的编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, newline='') as csvfile:
                    # 自动检测分隔符
                    sample = csvfile.read(1024)
                    csvfile.seek(0)
                    
                    # 检测分隔符
                    delimiter = ','
                    if '\t' in sample:
                        delimiter = '\t'
                    elif ';' in sample:
                        delimiter = ';'
                    
                    reader = csv.reader(csvfile, delimiter=delimiter)
                    rows = list(reader)
                    
                    if not rows:
                        return None, f"CSV文件为空"
                    
                    headers = rows[0]
                    data_rows = rows[1:] if len(rows) > 1 else []
                    
                    # 处理空值
                    processed_rows = []
                    for row in data_rows[:20]:  # 只取前10行
                        processed_row = [cell.strip() if cell else '' for cell in row]
                        # 确保行长度与header一致
                        while len(processed_row) < len(headers):
                            processed_row.append('')
                        processed_rows.append(processed_row[:len(headers)])
                    
                    return {
                        'headers': [h.strip() for h in headers],
                        'rows': processed_rows,
                        'total_rows': len(rows) - 1,  # 减去header行
                        'total_columns': len(headers),
                        'encoding': encoding
                    }, None
                    
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"使用编码 {encoding} 解析失败: {e}")
                continue
        
        return None, "无法解析CSV文件，可能的编码问题"
        
    except Exception as e:
        return None, f"CSV解析错误: {str(e)}"

def fix_html_content(content):
    """修复HTML内容中的常见问题"""
    import re
    
    # 1. 移除可能的语法错误标记
    content = re.sub(r'<<+[^>]*>>+', '', content)
    
    # 2. 检查并修复库引用问题
    libraries_to_add = []
    
    # 检查Plotly
    if 'Plotly' in content:
        if 'plotly' not in content.lower() or 'cdn.plot.ly' not in content:
            libraries_to_add.append({
                'name': 'Plotly',
                'script': '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
                'backup': '<script src="https://unpkg.com/plotly.js@2.26.0/dist/plotly.min.js"></script>'
            })
    
    # 检查Chart.js
    if 'Chart' in content and ('new Chart' in content or 'Chart.' in content):
        if 'chart.js' not in content.lower():
            libraries_to_add.append({
                'name': 'Chart.js',
                'script': '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>',
                'backup': '<script src="https://unpkg.com/chart.js@4.4.0/dist/chart.min.js"></script>'
            })
    
    # 检查D3.js
    if 'd3.' in content or 'd3[' in content:
        if 'd3js.org' not in content.lower():
            libraries_to_add.append({
                'name': 'D3.js',
                'script': '<script src="https://d3js.org/d3.v7.min.js"></script>',
                'backup': '<script src="https://unpkg.com/d3@7.8.5/dist/d3.min.js"></script>'
            })
    
    # 3. 添加库和错误处理
    if libraries_to_add:
        # 创建库加载脚本
        lib_scripts = []
        error_handling = []
        
        for lib in libraries_to_add:
            lib_scripts.append(lib['script'])
            lib_scripts.append(lib['backup'])  # 添加备用CDN
            
            # 添加库检测代码
            if lib['name'] == 'Plotly':
                error_handling.append('''
        // Plotly加载检测和备用方案
        function checkPlotly() {
            if (typeof Plotly === 'undefined') {
                console.error('Plotly未加载，尝试备用方案...');
                var script = document.createElement('script');
                script.src = 'https://unpkg.com/plotly.js@2.26.0/dist/plotly.min.js';
                script.onload = function() {
                    console.log('Plotly备用源加载成功');
                    if (typeof initPlotlyCharts === 'function') {
                        initPlotlyCharts();
                    }
                };
                script.onerror = function() {
                    console.error('Plotly备用源也加载失败');
                    document.body.innerHTML += '<div style="background:#ffebee;color:#c62828;padding:15px;margin:10px;border-radius:5px;"><strong>错误:</strong> 无法加载Plotly图表库</div>';
                };
                document.head.appendChild(script);
            } else {
                console.log('Plotly加载成功');
            }
        }''')
            
            elif lib['name'] == 'Chart.js':
                error_handling.append('''
        // Chart.js加载检测
        function checkChart() {
            if (typeof Chart === 'undefined') {
                console.error('Chart.js未加载');
                document.body.innerHTML += '<div style="background:#ffebee;color:#c62828;padding:15px;margin:10px;border-radius:5px;"><strong>错误:</strong> 无法加载Chart.js图表库</div>';
            } else {
                console.log('Chart.js加载成功');
            }
        }''')
        
        # 创建完整的头部脚本
        head_content = f'''
    <meta charset="UTF-8">
    <title>数据图表</title>
    <!-- 图表库 -->
    {chr(10).join(lib_scripts)}
    
    <!-- 错误处理和库检测 -->
    <script>
        // 全局错误处理
        window.addEventListener('error', function(e) {{
            console.error('JavaScript错误:', e.message, 'at line', e.lineno);
            var errorDiv = document.createElement('div');
            errorDiv.style.cssText = 'background:#ffebee;color:#c62828;padding:10px;margin:10px;border-radius:5px;border-left:4px solid #f44336;';
            errorDiv.innerHTML = '<strong>JavaScript错误:</strong> ' + e.message;
            document.body.insertBefore(errorDiv, document.body.firstChild);
            return true;
        }});
        
        // 未处理的Promise错误
        window.addEventListener('unhandledrejection', function(e) {{
            console.error('Promise错误:', e.reason);
            e.preventDefault();
        }});
        
        {chr(10).join(error_handling)}
        
        // 页面加载完成后检查库
        window.addEventListener('DOMContentLoaded', function() {{
            setTimeout(function() {{
                {'checkPlotly();' if any('Plotly' in lib['name'] for lib in libraries_to_add) else ''}
                {'checkChart();' if any('Chart' in lib['name'] for lib in libraries_to_add) else ''}
                
                // 延迟执行用户脚本
                setTimeout(function() {{
                    // 触发自定义初始化事件
                    var event = new CustomEvent('librariesLoaded');
                    document.dispatchEvent(event);
                }}, 1000);
            }}, 500);
        }});
    </script>
    
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
            line-height: 1.6;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 10px 0;
        }}
        .error-message {{
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #f44336;
        }}
        .success-message {{
            background: #e8f5e8;
            color: #2e7d2e;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #4caf50;
        }}
    </style>'''
        
        # 添加到HTML中
        if '<head>' in content:
            if '</head>' in content:
                content = content.replace('</head>', f'{head_content}\n</head>')
            else:
                content = content.replace('<head>', f'<head>\n{head_content}')
        else:
            # 没有head标签，添加完整的head部分
            if '<html>' in content:
                content = content.replace('<html>', f'<html>\n<head>\n{head_content}\n</head>')
            else:
                content = f'<head>\n{head_content}\n</head>\n{content}'
    
    # 4. 修复用户脚本中的Plotly调用
    if 'Plotly' in content:
        # 将直接的Plotly调用包装在检查函数中
        plotly_calls = re.findall(r'(Plotly\.[^;]+;)', content)
        for call in plotly_calls:
            safe_call = f'''
if (typeof Plotly !== 'undefined') {{
    {call}
}} else {{
    console.error('Plotly未加载，无法执行: {call}');
    document.addEventListener('librariesLoaded', function() {{
        if (typeof Plotly !== 'undefined') {{
            {call}
        }}
    }});
}}'''
            content = content.replace(call, safe_call)
    
    # 5. 确保HTML有基本结构
    if not content.strip().startswith('<!DOCTYPE'):
        if '<html>' not in content:
            content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>数据图表</title>
</head>
<body>
    <div class="chart-container">
        {content}
    </div>
</body>
</html>'''
        else:
            content = f'<!DOCTYPE html>\n{content}'
    
    return content

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """处理CSV文件上传"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件被上传'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if file and allowed_file(file.filename):
            # 安全的文件名处理
            filename = secure_filename(file.filename)
            csv_name = filename.replace('.csv', '')
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 保存文件
            file.save(file_path)
            logger.info(f"CSV文件已保存: {file_path}")
            
            # 解析CSV数据
            csv_data, error = parse_csv_simple(file_path)
            
            if error:
                return jsonify({'error': error}), 400
            
            # 添加文件名
            csv_data['filename'] = csv_name
            
            logger.info(f"CSV解析成功: {csv_name}, 行数: {csv_data['total_rows']}, 列数: {csv_data['total_columns']}")
            return jsonify(csv_data)
        
        return jsonify({'error': '请上传CSV文件'}), 400
        
    except Exception as e:
        logger.error(f"上传处理错误: {str(e)}")
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/get_html/<csv_name>/<prefix>')
def get_html(csv_name, prefix):
    """获取指定前缀的HTML文件"""
    try:
        # 安全的文件名处理
        csv_name = secure_filename(csv_name)
        prefix = secure_filename(prefix)
        
        folder_path = os.path.join(app.config['HTML_FOLDER'], csv_name)
        print(f"正在搜索HTML文件: {folder_path}，前缀: {prefix}")
        if not os.path.exists(folder_path):
            logger.warning(f"文件夹不存在: {folder_path}")
            return jsonify({'error': f'文件夹不存在: own/{csv_name}/'}), 404
        
        # 搜索匹配的HTML文件
        pattern = os.path.join(folder_path, f'{prefix}_*.html')
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            # 尝试其他可能的文件名格式
            alternative_patterns = [
                os.path.join(folder_path, f'{prefix}.html'),
                os.path.join(folder_path, f'{prefix}_chart.html'),
                os.path.join(folder_path, f'{prefix}_graph.html'),
                os.path.join(folder_path, f'{prefix}_data.html'),
                os.path.join(folder_path, f'{prefix}_report.html'),
            ]
            
            for pattern in alternative_patterns:
                alt_files = glob.glob(pattern)
                if alt_files:
                    matching_files = alt_files
                    break
        
        if not matching_files:
            available_files = glob.glob(os.path.join(folder_path, '*.html'))
            available_names = [os.path.basename(f) for f in available_files]
            logger.warning(f"未找到匹配 {prefix}_*.html 的文件，可用文件: {available_names}")
            return jsonify({
                'error': f'未找到匹配 {prefix}_*.html 的文件',
                'available_files': available_names
            }), 404
        
        # 返回第一个匹配的文件
        html_file = matching_files[0]
        logger.info(f"找到HTML文件: {html_file}")
        
        try:
            # 尝试不同编码读取HTML文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
            content = None
            
            for encoding in encodings:
                try:
                    with open(html_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"无法读取HTML文件 {html_file}，尝试的编码: {encodings}")
                return jsonify({'error': 'HTML文件编码不支持'}), 500
            
            # 修复HTML内容
            # content = fix_html_content(content)
            
            return jsonify({
                'content': content,
                'filename': os.path.basename(html_file),
                'file_path': os.path.relpath(html_file)
            })
            
        except Exception as e:
            logger.error(f"读取HTML文件错误: {str(e)}")
            return jsonify({'error': f'读取文件错误: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"获取HTML文件错误: {str(e)}")
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'html_folder': app.config['HTML_FOLDER'],
        'dependencies': 'minimal (no pandas)'
    })

@app.errorhandler(413)
def too_large(e):
    """文件过大错误处理"""
    return jsonify({'error': '文件过大，请上传小于16MB的文件'}), 413

@app.errorhandler(404)
def not_found(e):
    """404错误处理"""
    return jsonify({'error': '请求的资源不存在'}), 404

@app.errorhandler(500)
def internal_error(e):
    """500错误处理"""
    return jsonify({'error': '服务器内部错误'}), 500


# @app.route('/run_analysis/<csv_name>', methods=['POST'])
# def run_analysis(csv_name):
#     """运行后端分析代码"""
#     try:
#         # 安全的文件名处理
#         csv_name = secure_filename(csv_name)
        
#         # 导入你的后端代码模块
#         import run  # 假设你的后端代码在 run.py 文件中
        
#         # 调用 main 函数
#         logger.info(f"开始运行分析: {csv_name}")
#         run.main(data_name=csv_name)
        
#         return jsonify({
#             'success': True,
#             'message': f'分析完成: {csv_name}',
#             'csv_name': csv_name
#         })
        
#     except Exception as e:
#         logger.error(f"运行分析错误: {str(e)}")
#         return jsonify({
#             'success': False,
#             'error': f'分析失败: {str(e)}'
#         }), 500



# 在现有路由后添加这个新路由
@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """运行后端数据分析"""
    try:
        data = request.get_json()
        data_name = data.get('data_name')
        
        if not data_name:
            return jsonify({'error': '缺少data_name参数'}), 400
        
        # 调用run.py脚本
        result = subprocess.run([
            sys.executable, 'run.py', 
            '--data_name', data_name
        ], capture_output=True, text=True, timeout=300)  # 5分钟超时
        
        if result.returncode == 0:
            return jsonify({
                'status': 'success',
                'message': '分析完成',
                'output': result.stdout,
                'data_name': data_name
            })
        else:
            return jsonify({
                'status': 'error',
                'error': result.stderr,
                'output': result.stdout
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({'error': '处理超时，请稍后重试'}), 408
    except Exception as e:
        logger.error(f"运行分析错误: {str(e)}")
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500


if __name__ == '__main__':
    print("🚀 启动CSV处理Demo应用 (简化版)")
    print(f"📁 上传目录: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 HTML目录: {app.config['HTML_FOLDER']}")
    print("🌐 访问地址: http://localhost:5100")
    print("⚡ 不依赖Pandas，使用Python内置csv模块")
    print("🔄 按 Ctrl+C 停止服务器")
    
    app.run(debug=True, host='0.0.0.0', port=5100)