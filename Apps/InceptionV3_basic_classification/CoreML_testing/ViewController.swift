//
//  ViewController.swift
//  CoreML_testing
//
//  Created by Brandon Kates on 3/23/19.
//  Copyright © 2019 Brandon Kates. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController, UINavigationControllerDelegate {
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var classifier: UILabel!
    
    let mainColor = UIColor(red: 88/255.0, green: 196/255.0, blue: 91/255.0, alpha: 1.0)
    let lightColor = UIColor(red: 215.0/255.0, green: 235.0/255.0, blue: 202.0/255.0, alpha: 1.0)
    
    var model: Inceptionv3!
    
    var appTitleLabel =  UILabel()
    var CDSLabel =  UILabel()
    var introductionText = UITextView()
    
    override func viewWillAppear(_ animated: Bool) {
        model = Inceptionv3()
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white
        
        appTitleLabel.translatesAutoresizingMaskIntoConstraints = false
        appTitleLabel.text = "Seedlings Classifier"
        appTitleLabel.backgroundColor = .white
        appTitleLabel.font = UIFont.systemFont(ofSize: 40, weight: .bold)
        appTitleLabel.textAlignment = .left
        appTitleLabel.textColor = .black
        view.addSubview(appTitleLabel)
        
        CDSLabel.translatesAutoresizingMaskIntoConstraints = false
        CDSLabel.text = "Cornell Data Science"
        CDSLabel.font = UIFont.systemFont(ofSize: 15, weight: .regular)
        CDSLabel.textAlignment = .left
        CDSLabel.backgroundColor = .white
        CDSLabel.textColor = mainColor
        view.addSubview(CDSLabel)
        
        var introduction: String
        introduction = "Welcome!\n- To take a photo, press on the camera icon on the top left.\n" +
        "- To choose a photo from your library, press on the Library button on the top right."
        introductionText.translatesAutoresizingMaskIntoConstraints = false
        introductionText.isEditable = false
        introductionText.font = UIFont.systemFont(ofSize: 15, weight: .regular)
        introductionText.textColor = .black
        introductionText.backgroundColor = .white
        introductionText.text = introduction
        view.addSubview(introductionText)
        
        setUpConstraints()
    }
    
    func setUpConstraints() {
        NSLayoutConstraint.activate([
            appTitleLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 30),
            appTitleLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            appTitleLabel.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            appTitleLabel.heightAnchor.constraint(equalToConstant: 45)
            ])
        NSLayoutConstraint.activate([
            CDSLabel.topAnchor.constraint(equalTo: appTitleLabel.bottomAnchor),
            CDSLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            CDSLabel.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            CDSLabel.heightAnchor.constraint(equalToConstant: 20)
            ])
        NSLayoutConstraint.activate([
            introductionText.topAnchor.constraint(equalTo: CDSLabel.bottomAnchor, constant: 20),
            introductionText.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            introductionText.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            introductionText.heightAnchor.constraint(equalToConstant: 100)
            ])
        NSLayoutConstraint.activate([
            classifier.topAnchor.constraint(equalTo: introductionText.bottomAnchor, constant: 40)
            ])
        
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    @IBAction func camera(_ sender: Any) {
        if !UIImagePickerController.isSourceTypeAvailable(.camera) {
            return
        }
        let cameraPicker = UIImagePickerController()
        cameraPicker.delegate = self
        cameraPicker.sourceType = .camera
        cameraPicker.allowsEditing = false
        cameraPicker.showsCameraControls = true
        present(cameraPicker, animated: true)
    }
    
    @IBAction func openLibrary(_ sender: Any) {
        let picker = UIImagePickerController()
        picker.allowsEditing = false
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }
    
}

extension ViewController: UIImagePickerControllerDelegate {
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
    
    
    @objc func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        picker.dismiss(animated: true)
        classifier.text = "Analyzing Image..."
        guard let image = info["UIImagePickerControllerOriginalImage"] as? UIImage else {
            return
        }
            
        print(image.size)
        
        UIGraphicsBeginImageContextWithOptions(CGSize(width: 299, height: 299), true, 2.0)
        image.draw(in: CGRect(x: 0, y: 0, width: 299, height: 299))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        print(newImage.size)
        
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(newImage.size.width), Int(newImage.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard (status == kCVReturnSuccess) else {
            return
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: Int(newImage.size.width), height: Int(newImage.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) //3
        
        context?.translateBy(x: 0, y: newImage.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context!)
        newImage.draw(in: CGRect(x: 0, y: 0, width: newImage.size.width, height: newImage.size.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        imageView.image = newImage
        //NSLayoutConstraint.activate([
            //imageView.topAnchor.constraint(equalTo: //introductionText.bottomAnchor, constant: 20),
            //])
        guard let prediction = try? model.prediction(image: pixelBuffer!) else {
            return
        }
        classifier.text = "I think this is a \(prediction.classLabel)."
    }
}
