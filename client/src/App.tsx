import "antd/dist/antd.css";
import "./App.css";

import { Button, Col, Form, Input, Row, Modal  } from "antd";
import axios from 'axios'
import { HOST_NAME } from "./config";
import { Footer, Header } from "./components";

interface IResponse<T>{
    response: T
}

interface IHouseResponse {
  bayesian: number
  en: number
  lasso: number
  ols: number
  ridge: number
  forest: number
}

function App() {

  const [form] = Form.useForm();

  const info = (info: any) => {    
    Modal.info({
      maskClosable: true,
      content: renderModalInfo(info)
    });
  };

  const onSubmit = async () => {
    try {
      const values = await form.validateFields();
      axios.post<IResponse<IHouseResponse>>(`${HOST_NAME}api/v1/predict`,values)
      .then(({data}) => {
        info(data.response)
      })
    } catch (errorInfo) {
      console.log('Failed:', errorInfo);
    }
  
  }

  const getRequiredError = () =>  ([
        {
          required: true,
          message: 'სავალდებულო ველია',
        },
  ])
  

  const renderModalInfo = (house: IHouseResponse): JSX.Element => {
    return  <div className="center">
        <p>OLS: {Math.round(house.bayesian)} $</p>
        <p>Ridge: {Math.round(house.en)} $</p>
        <p>Lasso: {Math.round(house.lasso)} $</p>
        <p>Bayesian: {Math.round(house.ols)} $</p>
        <p>ElasticNet: {Math.round(house.ridge)} $</p>
        <p>Forest: {Math.round(house.forest)} $</p>
    </div>
  }
  return (
    <>
      <Header />
      <Row
        style={{
          justifyContent: "center",
          height: "calc(100vh - 100px)",
          paddingTop: "50px",
        }}
      >
        <Col span={12}>
          <Form
            labelCol={{
              span: 4,
            }}
            wrapperCol={{
              span: 14,
            }}
            layout="horizontal"
            initialValues={{
              size: "large",
            }}
            size={"large"}
            form={form} 
          >
            <Row>
              <Col span={12}>
                <Form.Item rules={getRequiredError()} name="latitude">
                  <Input type='number' placeholder="განედი"/>
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item rules={getRequiredError()} name="longitude">
                  <Input type='number' placeholder="გრძედი" />
                </Form.Item>
              </Col>
            </Row>

            <Row>
              <Col span={12}>
                <Form.Item rules={getRequiredError()} name="balcony">
                  <Input type='number' placeholder="ბალკონის რაოდენობა" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item rules={getRequiredError()} name="numberOfBathrooms">
                  <Input type='number' placeholder="სველი წერტილები" />
                </Form.Item>
              </Col>
            </Row>

            <Row>
              <Col span={12}>
                <Form.Item rules={getRequiredError()} name="loggia">
                  <Input type='number' placeholder="ლოჯის რაოდენობა" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item rules={getRequiredError()} name="veranda">
                  <Input type='number' placeholder="ვერანდას რაოდენობა" />
                </Form.Item>
              </Col>
            </Row>

            <Row>
              <Col span={12}>
                <Form.Item rules={getRequiredError()} name="kitchen">
                  <Input type='number' placeholder="სამზარეულოს რაოდენობა" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item rules={getRequiredError()} name="numberOfFloors">
                  <Input type='number' placeholder="სართულების რაოდენობა(ჯამი)" />
                </Form.Item>
              </Col>
            </Row>

            <Row>
              <Col span={12}>
                <Form.Item rules={getRequiredError()} name="floor">
                  <Input type='number' placeholder="სართული" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item rules={getRequiredError()} name="numberOfRooms">
                  <Input type='number' placeholder="ოთახების რაოდენობა" />
                </Form.Item>
              </Col>
            </Row>

            <Row>
              <Col span={12}>
                <Form.Item rules={getRequiredError()} name="LotArea">
                  <Input type='number' placeholder="ფართობი(კვ.მ)" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item rules={getRequiredError()} name="numberOfBedrooms">
                  <Input type='number' placeholder="საწოლი ოთახების რაოდენობა" />
                </Form.Item>
              </Col>
            </Row>

            <Row>
              <Col span={12}>
                <Form.Item rules={getRequiredError()} name="ceilingHeight">
                  <Input type='number' placeholder="ჭერის ზომა" />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item>
              <Button onClick={onSubmit} type='primary'>გამოთვლა</Button>
            </Form.Item>
          </Form>
        </Col>
      </Row>
      <Footer />
    </>
  );
}

export default App;
