import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Classificação de Áudio',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  static const _channel = MethodChannel('com.example.audio/audio_processor');
  String _result = ''; // Armazena o resultado das métricas
  String _executionTime = ''; // Armazena o tempo de execução do método

  Future<void> _executeAlgorithm(String algorithm) async {
    final stopwatch = Stopwatch()..start(); // Inicia o cronômetro

    try {
      String result = await _channel.invokeMethod(algorithm);
      stopwatch.stop(); // Para o cronômetro após a execução
      setState(() {
        _result = result; // Atualiza o estado com o resultado
        _executionTime = 'Tempo de execução: ${stopwatch.elapsedMilliseconds} ms';
      });
    } catch (e) {
      stopwatch.stop(); // Garante que o cronômetro pare mesmo em caso de erro
      setState(() {
        _result = ""; // Atualiza o estado com um mapa vazio em caso de erro
        _executionTime = 'Erro ao executar o método.';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Classificação de Áudio')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () => _executeAlgorithm('onnxr'),
              child: Text('Executar ONNX'),
            ),
            SizedBox(height: 20),
            Text(
              _result, // Exibe as métricas formatadas
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 20),
            Text(
              _executionTime, // Exibe o tempo de execução
              style: TextStyle(fontSize: 16, color: Colors.grey[700]),
            ),
          ],
        ),
      ),
    );
  }

}